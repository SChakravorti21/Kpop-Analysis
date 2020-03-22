import os
import utils
import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.stat import Correlation


TRACK_FOLDERS = "data/*pop/*.json"


def main(spark: SparkSession):
    df = spark.read.json(TRACK_FOLDERS, multiLine=True)
    
    vector_mapper = udf(lambda row: Vectors.dense(row), VectorUDT())
    df = df.withColumn("features", vector_mapper(array("loudness", "energy")))
    
    r1 = Correlation.corr(df, "features").head()
    print(r1[0])

    pairs = df.select("loudness", "energy").collect()
    x = [pair["loudness"] for pair in pairs]
    y = [pair["energy"]   for pair in pairs]

    plt.figure(figsize=(16, 12))
    plt.scatter(x, y, marker='o', alpha=0.3)
    plt.xlabel("loudness")
    plt.ylabel("energy")
    plt.title("Comparing loudness to energy")
    plt.savefig(os.path.join("analysis", "results", "loudness_energy_corr.png"))
    plt.close()


if __name__ == "__main__":
    main(SparkSession
            .builder
            .master("local[*]")
            .appName("Loudness/Energy Correlation")
            .getOrCreate())