import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from analysis_prelim import FEATURE_KEYS


TRACK_FEATURES = os.path.join("data", "*pop-track-features.json")


if __name__ == "__main__":
    spark = SparkSession \
                .builder \
                .appName("Pop/Kpop Analysis") \
                .master("local[*]") \
                .getOrCreate()

    # UDF we will use to add feature column to DataFrame
    vector_mapper = udf(lambda row: Vectors.dense(row), VectorUDT())

    # Load both pop and kpop data
    dataset = spark.read.json(TRACK_FEATURES, multiLine=True)
    dataset = dataset.withColumn("features", vector_mapper(array(*FEATURE_KEYS)))
    silhouettes = []

    for iteration, k in enumerate(range(2, 10)):
        # Define the model
        kmeans = KMeans(k=k, seed=17386423)
        model = kmeans.fit(dataset)

        # Make predictions; we are going to predict straight on our
        # training dataset since the clustering was derived from it
        predictions = model.transform(dataset)
        
        # Compute error
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        silhouettes.append(silhouette)

    sns.lineplot(x=list(range(2, 10)), y=silhouettes, palette="coolwarm")
    plt.savefig(os.path.join("analysis", "results", "charts", "k-means-elbow.png"))

