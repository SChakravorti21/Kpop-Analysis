import os
import utils
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, array
from pyspark.ml.linalg import Vectors, VectorUDT, DenseMatrix
from pyspark.ml.stat import Correlation
from analysis_prelim import FEATURE_KEYS


TRACK_FEATURES = os.path.join("data", "*pop-track-features.json")


def save_corr_heatmap(corr: DenseMatrix, columns: List[str],
                      path: str, title="Correlation Matrix"):
    rows = corr.toArray().tolist()
    df = pd.DataFrame(rows)
    fig = plt.figure(figsize=(14, 10))
    sns.heatmap(df, xticklabels=columns, yticklabels=columns, annot=True)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def find_correlation(spark: SparkSession, to_correlate: List[str],
                     features_file=TRACK_FEATURES):
    df = spark.read.json(features_file, multiLine=True)
    vector_mapper = udf(lambda row: Vectors.dense(row), VectorUDT())

    df = df.withColumn("features", vector_mapper(array(*to_correlate)))
    corr_matrix = Correlation.corr(df, "features").head()[0]
    return df, corr_matrix


def loudness_energy_corr(spark: SparkSession):
    plot_path = os.path.join("analysis", "results", 
                             "charts", "scatter-loudness-energy.png")
    corr_path = os.path.join("analysis", "results", 
                             "charts", "corr-loudness-energy.png")

    features = ["loudness", "energy"]
    df, corr = find_correlation(spark, features)
    save_corr_heatmap(corr, features, corr_path)

    pairs = df.select(*features).collect()
    x = [pair["loudness"] for pair in pairs]
    y = [pair["energy"]   for pair in pairs]

    plt.figure(figsize=(16, 12))
    plt.scatter(x, y, marker='o', alpha=0.5)
    plt.xlabel("loudness")
    plt.ylabel("energy")
    plt.title("Comparing loudness to energy")
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    spark = SparkSession \
                .builder \
                .master("local[*]") \
                .appName("Loudness/Energy Correlation") \
                .getOrCreate()
    
    loudness_energy_corr(spark)

    for genre in ("kpop", "pop"):
        features_file = os.path.join("data", f"{genre}-track-features.json")
        _, corr = find_correlation(spark, FEATURE_KEYS, features_file)
        corr_path = os.path.join("analysis", "results", 
                                 "charts", f"corr-all-{genre}.png")
        save_corr_heatmap(corr, FEATURE_KEYS, corr_path)