import os
import io
import sys
import enum
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Union
from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import FloatType
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

# Seed KMeans operations for reproducibility
SEED           = 17386423
# UDF we can use to add feature column to DataFrame
VECTOR_MAPPER  = udf(lambda row: Vectors.dense(row), VectorUDT())
POP_TRACKS     = os.path.join("data", "pop-tracks.json")
KPOP_TRACKS    = os.path.join("data", "kpop-tracks.json")
KPOP_TRACKS_LG = os.path.join("data", "kpop-tracks-lg.json")


class DatasetComposition(enum.Enum):
    MIXED_POP_KPOP  = 0
    ONLY_KPOP       = 1


class ClusterAnalyzer():
    def __init__(self, kind: DatasetComposition, k: int,
                 kmeans_type: Union[KMeans, BisectingKMeans]):
        if kind == DatasetComposition.MIXED_POP_KPOP:
            self.tracks         = os.path.join("data", "*pop-track-features.json")
            self.features       = FEATURE_KEYS
            self.dataset_name   = "general"
        else:
            self.tracks         = os.path.join("data", "kpop-track-features-lg.json")
            self.features       = FEATURE_KEYS
            self.dataset_name   = "kpop"

        self.k = k
        self.dataset_type = kind
        self.kmeans_type  = kmeans_type
        self.kmeans_name  = ("k-means"
                             if kmeans_type == KMeans
                             else "bisect-k-means")
        self.model_name   = f"{self.dataset_name}-{self.kmeans_name}-{k}"
        self.model_path   = os.path.join("analysis", "models", self.model_name)

        self.spark = SparkSession \
            .builder \
            .appName("Pop/Kpop Analysis") \
            .master("local[*]") \
            .getOrCreate()

        # Removing songs that are most likely live (i.e. performed
        # at concerts, tours, etc.) because they only add
        # noise to our specific analyses
        self.dataset = self.spark \
            .read.json(self.tracks, multiLine=True) \
            .filter(col("liveness") < 0.70) \
            .withColumn("features", VECTOR_MAPPER(array(*self.features))) \
            .persist(StorageLevel.MEMORY_ONLY) \

    def analyze_clusters(self):
        if self.dataset_type == DatasetComposition.MIXED_POP_KPOP:
            self._analyze_general()
        else:
            self._analyze_kpop()

    def _analyze_general(self):
        # Figure out which songs belong to which clusters
        kmeans_model = self._load_kmeans_model()
        dataset = kmeans_model.transform(self.dataset).cache()

        # Show the parameters of each cluster center
        self._print_cluster_stats(kmeans_model)

        # Now we need to figure out which songs are pop vs. k-pop
        pop_tracks  = self.spark.read.json(POP_TRACKS, multiLine=True) \
                        .withColumn("genre-pop", lit(1)) \
                        .withColumn("genre-kpop", lit(0)) \
                        .select(
                            "id", "genre-pop", "genre-kpop",
                            "name", "popularity",
                            col("external_urls.spotify").alias("track-url"),
                            col("artists.name").alias("track-artists"))
        kpop_tracks = self.spark.read.json(KPOP_TRACKS, multiLine=True) \
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

    def _analyze_kpop(self):
        # Figure out which songs belong to which clusters
        kmeans_model = self._load_kmeans_model()
        predictions  = kmeans_model.transform(self.dataset).cache()
        total_tracks = predictions.count()

        # Show the parameters of each cluster center
        self._print_cluster_stats(kmeans_model)

        # Get track details
        kpop_tracks = self.spark \
            .read.json(KPOP_TRACKS_LG, multiLine=True) \
            .select(
                "id", "name",
                col("external_urls.spotify").alias("track-url"),
                col("artists.name").alias("track-artists")) \
            .join(predictions, "id") \
            .persist(StorageLevel.MEMORY_ONLY) \

        # Get the cluster centers so we can later get samples which
        # are representative of the cluster. Consume numpy array
        # since Spark can't create DataFrame out of it
        centers = [center.tolist() for center in kmeans_model.clusterCenters()]
        centers = zip(range(len(centers)), centers)
        centers = self.spark.createDataFrame(centers, ["clusterNum", "clusterCenter"])

        # Show largest clusters first
        cluster_info = predictions \
            .groupBy("clusterNum") \
            .count() \
            .withColumnRenamed("count", "numTracks") \
            .join(centers, "clusterNum") \
            .orderBy(col("numTracks").desc()) \
            .collect()

        # Print size and sample songs for each cluster
        for cluster in cluster_info:
            center      = cluster["clusterCenter"]
            cluster_num = cluster["clusterNum"]
            num_tracks  = cluster["numTracks"]
            rel_size    = num_tracks / total_tracks * 100

            # Going to sort by distance from cluster center
            # to get the most representative songs
            sort_func = lambda row: utils.euclid_dist(row, center)
            sort_func = udf(sort_func, FloatType())

            sample_songs = kpop_tracks \
                .filter(col("clusterNum") == cluster_num) \
                .orderBy(sort_func("features")) \
                .take(15)

            print(f"{cluster_num:<2}: {num_tracks} tracks ({rel_size:.2f}%)")

            for song in sample_songs:
                print(f"\t{song['name']}, {', '.join(song['track-artists'])}\n\t\t({song['track-url']})")

            print()

    def plot_clusters(self):
        # Load the KMeans (or BisectingKMeans) model and derive
        # the cluster each song belong to.
        # The transformation simply adds a clusterNum column
        # to the DF, so we can pass this to PCA model as well
        dataset = self._load_kmeans_model().transform(self.dataset)

        # Perform dimensionality reduction to be able to see clusters
        # in two or three dimensions
        self._perform_pca(dataset, 2)
        self._perform_pca(dataset, 3)

    def train_and_save_model(self):
        kmeans = self.kmeans_type(k=self.k, seed=SEED,
            predictionCol="clusterNum")
        model = kmeans.fit(self.dataset)

        # Save the model to disk for later usage
        model_path = os.path.join("analysis", "models", self.model_name)
        model.write().overwrite().save(model_path)

    def find_elbow(self):
        x, y = [], []

        for iteration, k in enumerate(range(2, 50)):
            # Define the model, seed should be fixed between iteration
            # to prevent it from being a source of variance
            kmeans = self.kmeans_type(k=k, seed=SEED)
            model = kmeans.fit(self.dataset)

            # Make predictions; we are going to predict straight on our
            # training dataset since the clustering was derived from it
            predictions = model.transform(self.dataset)

            # Compute error
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)

            x.append(iteration)
            y.append(silhouette)

        sns.lineplot(x=x, y=y, palette="coolwarm")
        plot_name = f"elbow-{self.dataset_name}-{self.kmeans_name}.png"
        plt.savefig(os.path.join("analysis", "results", "charts", plot_name))

    def _perform_pca(self, dataset: DataFrame, k: int):
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
                                  "charts", f"pca-{k}-{self.model_name}.png")
        plt.savefig(image_path)

    def _load_kmeans_model(self) -> KMeansModel:
        """
        Although a BisectingKMeansModel isn't directly related
        to a KMeansModel, they have pretty much the same interface,
        and can therefore be used similarly.
        """

        model_type = (BisectingKMeansModel
                      if "bisect-k" in self.model_path
                      else KMeansModel)
        return model_type.load(self.model_path)

    def _print_cluster_stats(self, kmeans_model: Union[KMeansModel, BisectingKMeansModel]):
        # Print out the characteristics of each cluster so we can
        # examine what makes each cluster unique
        wrapper_df     = pd.DataFrame(kmeans_model.clusterCenters(), columns=self.features)
        string_wrapper = io.StringIO()
        wrapper_df.to_csv(string_wrapper, sep='\t', float_format="%.6f")
        table          = string_wrapper.getvalue()
        print(table, "\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "bisect":
        estimator = BisectingKMeans
    else:
        estimator = KMeans

    k = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    analyzer = ClusterAnalyzer(DatasetComposition.ONLY_KPOP, k, estimator)

    if sys.argv[1] == "elbow":
        analyzer.find_elbow()
    elif sys.argv[1] == "train":
        analyzer.train_and_save_model()
    elif sys.argv[1] == "plot":
        analyzer.plot_clusters()
    elif sys.argv[1] == "analyze":
        analyzer.analyze_clusters()
