import os
import io
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    udf, array, lit, 
    first, col, 
    get_json_object)
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import (
    KMeans, 
    BisectingKMeans, 
    KMeansModel, 
    BisectingKMeansModel)
from pyspark.ml.evaluation import ClusteringEvaluator
from analysis_prelim import FEATURE_KEYS


# UDF we can use to add feature column to DataFrame
SEED           = 17386423
VECTOR_MAPPER  = udf(lambda row: Vectors.dense(row), VectorUDT())
TRACK_FEATURES = os.path.join("data", "*pop-track-features.json")
POP_TRACKS     = os.path.join("data", "pop-tracks.json")
KPOP_TRACKS    = os.path.join("data", "kpop-tracks.json")


def load_kmeans_model(model_path: str) -> KMeansModel:
    """
    Although a BisectingKMeansModel isn't directly related
    to a KMeansModel, they have pretty much the same interface,
    and can therefore be used similarly.
    """

    model_type = (BisectingKMeansModel 
                 if "bisect-k" in model_path 
                 else KMeansModel)
    return model_type.load(model_path)


def analyze_clusters(spark: SparkSession, dataset: DataFrame, 
                     model_path: str):
    # Figure out which songs belong to which clusters
    kmeans_model = load_kmeans_model(model_path)
    dataset = kmeans_model.transform(dataset)

    # Now we need to figure out which songs are pop vs. k-pop
    pop_tracks  = spark.read.json(POP_TRACKS, multiLine=True) \
                    .withColumn("genre-pop", lit(1)) \
                    .withColumn("genre-kpop", lit(0)) \
                    .select(
                        "id", "genre-pop", "genre-kpop", 
                        "name", "popularity",
                        col("external_urls.spotify").alias("track-url"),
                        col("artists.name").alias("track-artists"))
    kpop_tracks = spark.read.json(KPOP_TRACKS, multiLine=True) \
                    .withColumn("genre-pop", lit(0)) \
                    .withColumn("genre-kpop", lit(1)) \
                    .select(
                        "id", "genre-pop", "genre-kpop", 
                        "name", "popularity",
                        col("external_urls.spotify").alias("track-url"),
                        col("artists.name").alias("track-artists"))
    
    # Associate each song with its genre
    # (obviously this is a slightly convoluted way of doing this,
    # but it gives me some practice doing joins with DataFrames)
    pop_subset  = dataset.join(pop_tracks, "id").cache()
    kpop_subset = dataset.join(kpop_tracks, "id").cache()

    # Seems like despite trying to avoid overlap in artists
    # between the two genres, we have double-counted some
    # songs which fall under both genres, so must use `distinct`
    dataset     = pop_subset.unionByName(kpop_subset).distinct()

    # For each cluster, figure out how many pop vs. kpop
    # songs it contains
    cluster_info = dataset \
        .groupBy("clusterNum") \
        .agg(F.sum("genre-pop").alias("pop"),
             F.sum("genre-kpop").alias("kpop")) \
        .withColumn("total", col("pop") + col("kpop")) \
        .withColumn("perc-pop", col("pop") / col("total") * 100) \
        .withColumn("perc-kpop", col("kpop") / col("total") * 100) \
        .collect()

    # Not entirely sure how to do this kind of sorting directly
    # on the Spark DataFrame, so doing it in plain Python instead
    # (lists are pretty small anyways, never more than 30 resulting rows)
    cluster_info.sort(
        reverse=True,
        key=lambda cluster: max(cluster["perc-pop"], cluster["perc-kpop"])
    )

    # Print out the characteristics of each cluster so we can
    # examine what makes each cluster unique
    wrapper_df     = pd.DataFrame(kmeans_model.clusterCenters(), columns=FEATURE_KEYS)
    string_wrapper = io.StringIO()
    wrapper_df.to_csv(string_wrapper, sep='\t', float_format="%.6f")
    table          = string_wrapper.getvalue()
    print(table, "\n")

    # Print out what percent of each cluster is pop vs. kpop,
    # plus offer some sample songs for diving a bit deeper
    print("Cluster #\tPop / %\t\t\tKpop / %")
    for cluster in cluster_info:
        clusterNum  = cluster["clusterNum"]
        pop         = cluster["pop"]
        kpop        = cluster["kpop"]
        perc_pop    = cluster["perc-pop"]
        perc_kpop   = cluster["perc-kpop"]

        # Sample some of the most popular songs for this cluster
        pop_songs  = pop_subset \
            .filter(col("clusterNum")  == clusterNum) \
            .orderBy(col("popularity").desc()) \
            .take(3)
        kpop_songs = kpop_subset \
            .filter(col("clusterNum")  == clusterNum) \
            .orderBy(col("popularity").desc()) \
            .take(3)

        print(f"{clusterNum:<2}\t\t\t{pop:<2} / {perc_pop:.2f}%\t\t{kpop:<2} / {perc_kpop:.2f}%")

        for song in pop_songs:
            print(f"\tPop:  {song['name']}, {', '.join(song['track-artists'])}\n\t\t({song['track-url']})")
        for song in kpop_songs:
            print(f"\tKpop: {song['name']}, {', '.join(song['track-artists'])}\n\t\t({song['track-url']})")

        print()


def perform_pca(dataset: DataFrame, k: int, model_name: str):
    # Since we want to plot the clusters, it is important
    # downsize the dimensions to at most 3 dimensions.
    # We can use PCA with 3 principal components for this.
    pca = PCA(k=k, inputCol="features", outputCol="pcaFeatures")
    pca_model = pca.fit(dataset)
    rows = pca_model \
                .transform(dataset) \
                .select("clusterNum", "pcaFeatures") \
                .collect()

    # Now we'll plot the clusters as a 3D scatter plot with
    # each point's color corresponding to its cluster.
    # Cast cluterNum to string so it is treated as categorical
    # data for plotting purposes.
    axes = zip(*[row["pcaFeatures"] for row in rows])
    colors  = pd.Categorical([row["clusterNum"] for row in rows])

    if k == 2:
        x, y = axes
        fig = plt.figure(figsize=(15, 15))
        sns.scatterplot(x=x, y=y, hue=colors)
    if k == 3:
        x, y, z = axes
        plot_df = pd.DataFrame({"PCA 1": x, "PCA 2": y, "PCA 3": z, "cluster": colors})
        g = sns.PairGrid(plot_df, hue="cluster", palette="coolwarm")
        g = g.map(sns.scatterplot, linewidths=0.75, edgecolor="w", s=40)
        g = g.add_legend()
        g.fig.set_size_inches(15, 15)

    # Specify number of principal components and clusters in model
    image_path = os.path.join("analysis", "results", 
                              "charts", f"pca-{k}-{model_name}.png")
    plt.savefig(image_path)


def plot_clusters(dataset: DataFrame, model_path: str):
    # Load the KMeans (or BisectingKMeans) model and derive
    # the cluster each song belong to.
    # The transformation simply adds a clusterNum column
    # to the DF, so we can pass this to PCA model as well
    dataset = load_kmeans_model(model_path).transform(dataset)

    # The model path is technically a directory, so we need
    # to do this to get the final directory's name
    model_name = os.path.basename(os.path.dirname(model_path))
    perform_pca(dataset, 2, model_name)
    perform_pca(dataset, 3, model_name)


def train_and_save_model(dataset: DataFrame, estimator: Callable, 
                         k: int, model_path: str):
    kmeans = estimator(k=k, seed=SEED, predictionCol="clusterNum")
    model = kmeans.fit(dataset)
    model.write().overwrite().save(model_path)


def find_elbow(dataset: DataFrame, estimator: Callable, estimator_name: str):
    x, y = [], []

    for iteration, k in enumerate(range(2, 50)):
        # Define the model, seed should be fixed between iteration
        # to prevent it from being a source of variance
        kmeans = estimator(k=k, seed=SEED)
        model = kmeans.fit(dataset)

        # Make predictions; we are going to predict straight on our
        # training dataset since the clustering was derived from it
        predictions = model.transform(dataset)

        # Compute error
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)

        x.append(iteration)
        y.append(silhouette)

    sns.lineplot(x=x, y=y, palette="coolwarm")
    plt.savefig(os.path.join("analysis", "results",
                             "charts", f"elbow-{estimator_name}.png"))


if __name__ == "__main__":
    spark = SparkSession \
                .builder \
                .appName("Pop/Kpop Analysis") \
                .master("local[*]") \
                .getOrCreate()

    # Load both pop and kpop data
    df = spark.read.json(TRACK_FEATURES, multiLine=True).cache()
    df = df.withColumn("features", VECTOR_MAPPER(array(*FEATURE_KEYS)))
    estimator, est_name = KMeans, "k-means"

    if len(sys.argv) > 2 and sys.argv[2] == "bisect":
        estimator, est_name = BisectingKMeans, "bisect-k-means"

    if sys.argv[1] == "elbow":
        find_elbow(df, estimator, est_name)
    elif sys.argv[1] == "train":
        k = int(sys.argv[3])
        model_output = os.path.join("analysis", "models", f"{est_name}-{k}")
        train_and_save_model(df, estimator, k, model_output)
    elif sys.argv[1] == "plot":
        plot_clusters(df, sys.argv[2])
    elif sys.argv[1] == "analyze":
        analyze_clusters(spark, df, sys.argv[2])