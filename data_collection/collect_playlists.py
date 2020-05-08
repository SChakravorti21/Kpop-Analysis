import os
import json
import utils
from time import sleep
from data_collection.collect_tracks import get_track_features

PLAYLISTS = {
    "555CSj8SPEaY4FMWWZhdq3": "Happy",
    "2mxWcjRgZ1aOFJODEG2kns": "Chill",
    "4Bt7rs6xjp29iZ4d6CKgOJ": "Sad/Sentimental",
    "2Obe8YMzxqJGBs34TlUBSs": "Bops",
    "4DJu5gkpoLxBiYJzsbC993": "Madness"
}

sp = utils.get_spotipy_instance()

def get_playlist_tracks():
    tracks = []

    # Get the tracks for each of the playlists
    # so that we can get their features afterwards
    for playlist_id, name in PLAYLISTS.items():
        playlist_items = sp.playlist_tracks(playlist_id)["items"]

        for item in playlist_items:
            track = item["track"]
            track["playlist"] = name  # attach label to track

            # Delete album and markets info since it clutters the 
            # JSON and makes it harder to skim through the data
            del track["album"]
            del track["available_markets"]
            tracks.append(track)

    filtered = []
    features = get_track_features(tracks)

    for track, features in zip(tracks, features):
        # In some rare occasions, features might not be available
        if features is None:
            name, playlist = track["name"], track["playlist"]
            print(f"Missing features for {name} ({playlist})")
            continue
        
        features["playlist"] = track["playlist"]
        filtered.append((track, features))

    # Each entry is a tuple of track info and features.
    # If we make each tuple its own iterable and zip them all
    # together, we'll get separate lists of all tracks and features.
    tracks, features = zip(*filtered)
    tracks_path = os.path.join("data", "playlist_tracks.json")
    features_path = os.path.join("data", "playlist_features.json")
    utils.write_json(tracks, tracks_path)
    utils.write_json(features, features_path)

if __name__ == "__main__":
    get_playlist_tracks()