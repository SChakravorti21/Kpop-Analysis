import os
import utils
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import pyspark.sql.types as T
from typing import List
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.stat import Summarizer
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql import SparkSession, DataFrame
from clustering import KPOP_TRACKS_LG, KPOP_FEATURES


COMPARE_FEATURES = ["acousticness", "danceability", "energy",
                    "speechiness", "valence", "tempo", "loudness"]
KPOP_ARTISTS     = os.path.join("data", "kpop-*g.json")


class ArtistAnalyzer():
    def __init__(self):
        self.spark = SparkSession \
            .builder \
            .appName("Artist Analysis") \
            .master("local[*]") \
            .getOrCreate()

    def summarize_artist_styles(self):
        # We need to use a `Summarizer` to be able to take
        # the average of a Vector-type column
        songs = self._generate_dataset() \
            .withColumn("artist", F.expr("artists[0].name")) \
            .groupBy("artist") \
            .agg(Summarizer.mean(F.col("features")).alias("average_song")) \
            .select("artist", "average_song") \
            .collect()

        # Only keep track of some of the most popular artists,
        # there's way too many to realistically compare all of them
        """
        dataset = self.spark \
            .read.json(KPOP_ARTISTS, multiLine=True) \
            .withColumnRenamed("name", "artist") \
            .select("artist", "popularity") \
            .join(songs, "artist") \
            .orderBy(F.col("popularity").desc()) \
            .collect()
        """

        for row in songs:
            self._save_radar_plot(
                row["artist"],
                # DenseVector -> numpy.ndarray -> List[float]
                row["average_song"].toArray().tolist()
            )

    def _generate_dataset(self) -> DataFrame:
        track_info = self.spark \
            .read.json(KPOP_TRACKS_LG, multiLine=True)
        dataset = self.spark \
            .read.json(KPOP_FEATURES, multiLine=True) \
            .filter(F.col("liveness") < 0.70) \
            .join(track_info, "id")

        vectorizer = VectorAssembler(
            inputCols=COMPARE_FEATURES,
            outputCol="unscaled"
        )

        scaler = MinMaxScaler(
            min=0.0, max=1.0,
            inputCol=vectorizer.getOutputCol(),
            outputCol="features"
        )

        pipeline   = Pipeline(stages=[vectorizer, scaler])
        model      = pipeline.fit(dataset)

        return model \
            .transform(dataset) \
            .persist(StorageLevel.MEMORY_ONLY)

    def _save_radar_plot(
        self,
        artist: str,
        song: List[float]
    ):
        angles = np.linspace(0, 2 * np.pi, len(song), endpoint=False)

        # Duplicate first element to close the loop
        song.append(song[0])
        angles = np.concatenate((angles, [angles[0]]))

        # Create the radar plot
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, song, "o-", linewidth=2)
        ax.fill(angles, song, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, COMPARE_FEATURES)
        ax.set_title(artist)
        ax.grid(True)

        # Persist the chart
        output_path = os.path.join(
            "analysis", "results", "artists",
            f"{artist}.png"
        )

        utils.makedirs(output_path)
        fig.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    analyzer = ArtistAnalyzer()
    analyzer.summarize_artist_styles()
