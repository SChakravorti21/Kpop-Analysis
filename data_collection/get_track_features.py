import os
import json
import utils
import spotipy


sp = utils.get_spotipy_instance()


def output_track_features(tracks, output_path):
    all_tracks = []

    for track_batch in utils.batches(tracks, 50):
        track_ids = [track["id"] for track in track_batch]
        track_features = sp.audio_features(track_ids)
        all_tracks += track_features

    # Originally saved track features in one file per
    # track, but found that it took too long for Spark
    # to go through all the files and parse them into
    # a DataFrame. Storing all track features in a single
    # file gives tremendously better performance.
    utils.write_json(all_tracks, output_path)


if __name__ == "__main__":
    for genre in ("kpop", "pop"):
        tracklist_file = os.path.join("data", f"{genre}-tracks.json") 

        with open(tracklist_file, "r") as f:
            tracks = json.load(f)

        output_path = os.path.join("data", f"{genre}-track-features.json")
        output_track_features(tracks, output_path)
