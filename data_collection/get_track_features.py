import os
import json
import utils
import spotipy


sp = utils.get_spotipy_instance()


def output_track_features(tracks, output_dir):
    for track_batch in utils.batches(tracks, 50):
        track_ids = [track["id"] for track in track_batch]
        track_features = sp.audio_features(track_ids)

        for track in track_features:
            track_file = os.path.join(output_dir, f"{track['id']}.json")
            utils.write_json(track, track_file)


if __name__ == "__main__":
    for genre in ("kpop", "pop"):
        tracklist_file = os.path.join("data", f"{genre}-tracks.json") 

        with open(tracklist_file, "r") as f:
            tracks = json.load(f)

        output_track_features(tracks, os.path.join("data", genre))
