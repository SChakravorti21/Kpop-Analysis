import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pyspark import StorageLevel
from pyspark.sql import SparkSession, DataFrame


FEATURE_KEYS = ["acousticness", "danceability", "energy", 
                "instrumentalness", "liveness", "loudness", 
                "speechiness", "tempo", "valence"]


def get_track_df(spark: SparkSession, track_files: List[str]):
    df = spark.read.json(track_files, multiLine=True)
    df.persist(storageLevel=StorageLevel.MEMORY_ONLY)
    return df


def write_stats(df: DataFrame, path: str):
    desc = df.describe(FEATURE_KEYS).collect()
    desc_json = [row.asDict() for row in desc]
    utils.write_json(desc_json, path)


def show_hist(dfs: Dict[str, DataFrame], feature: str):
    bound_min, bound_max = float("inf"), float("-inf")
    genre_series: Dict[str, List[float]] = {}
    
    for genre in dfs:
        feature_df = dfs[genre].select(feature).cache()

        rows = feature_df.collect()
        rows = [row[feature] for row in rows]
        genre_series[genre] = rows

        genre_min = feature_df.agg({ feature : "min" }).collect()[0][0]
        genre_max = feature_df.agg({ feature : "max" }).collect()[0][0]

        bound_min = min(bound_min, genre_min)
        bound_max = max(bound_max, genre_max)

    bins = np.linspace(bound_min, bound_max, 100)
    fig, ax = plt.subplots(figsize=(16, 12))

    for genre, series in genre_series.items():
        plt.hist(series, bins, alpha=0.5, label=genre)
    
    ax.set_xlabel(f"{feature} value")
    ax.set_ylabel("count")
    plt.title(feature)
    plt.legend(loc='upper right')

    fig_path = os.path.join("analysis", "charts", f"{feature}-comp.png")
    utils.makedirs(fig_path)

    fig.savefig(fig_path)
    plt.close(fig)


def main(spark: SparkSession):
    genre_dfs: Dict[str, DataFrame] = {}
    
    for genre in ("kpop", "pop"):
        genre_tracks = os.listdir(os.path.join("data", genre))
        track_files = [os.path.join("data", genre, track) 
                       for track in genre_tracks]
        
        # Construct a Spark DataFrame out of the tracks for this genre
        df = get_track_df(spark, track_files)
        genre_dfs[genre] = df

        # Get a basic overview of track features like liveness, acousticness, etc.
        stats_path = os.path.join("analysis", f"{genre}-overview.json")
        stats = write_stats(df, stats_path)


    for feature in FEATURE_KEYS:
        show_hist(genre_dfs, feature)


if __name__ == "__main__":
    main(SparkSession
            .builder
            .master("local")
            .appName("Pop/Kpop Basic Stats")
            .getOrCreate())

