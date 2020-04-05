import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, array
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
    # the cluster each song belong to
    model_type = (BisectingKMeansModel 
                 if "bisect-k" in model_path 
                 else KMeansModel)
    kmeans_model = model_type.load(model_path)
    # The transformation simply adds a clusterNum column
    # to the DF, so we can pass this to PCA model as well
    dataset = kmeans_model.transform(dataset)

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