import os
import pyspark
import utils
from typing import List

FEATURE_KEYS = ["acousticness", "danceability", "energy", 
                "instrumentalness", "liveness", "loudness", 
                "speechiness", "tempo", "valence"]


def get_stats(spark: pyspark.sql.SparkSession, track_files: List[str]):
    df = spark.read.json(track_files, multiLine=True)
    desc = df.describe(FEATURE_KEYS).collect()
    desc_json = [row.asDict() for row in desc]
    return desc_json


if __name__ == "__main__":
    spark = pyspark.sql.SparkSession \
                .builder \
                .master("local") \
                .appName("Pop/Kpop Basic Stats") \
                .getOrCreate()
    
    for genre in ("kpop", "pop"):
        genre_tracks = os.listdir(os.path.join("data", genre))
        track_files = [os.path.join("data", genre, track) 
                       for track in genre_tracks]
        stats = get_stats(spark, track_files)

        output_path = os.path.join("analysis", f"{genre}-prelim.json")
        utils.write_json(stats, output_path)