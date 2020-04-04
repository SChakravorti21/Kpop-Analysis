import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from analysis_prelim import FEATURE_KEYS
from typing import Callable


# UDF we can use to add feature column to DataFrame
VECTOR_MAPPER  = udf(lambda row: Vectors.dense(row), VectorUDT())
TRACK_FEATURES = os.path.join("data", "*pop-track-features.json")


def find_elbow(spark: SparkSession, dataset: DataFrame, 
               estimator: Callable, estimator_name: str):
    dataset = dataset.withColumn("features", VECTOR_MAPPER(array(*FEATURE_KEYS)))
    x, y = [], []

    for iteration, k in enumerate(range(2, 50)):
        # Define the model, seed should be fixed between iteration
        # to prevent it from being a source of variance
        kmeans = estimator(k=k, seed=17386423)
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
    estimator, est_name = KMeans, "k-means"

    if len(sys.argv) > 2 and sys.argv[2] == "bisect":
        estimator, est_name = BisectingKMeans, "bisect-k-means"

    if sys.argv[1] == "elbow":
        find_elbow(spark, df, estimator, est_name)

