import os
import json
import spotipy
import utils
from spotipy.oauth2 import SpotifyClientCredentials


sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


def get_popular_tracks(artists_file):
    with open(artists_file, "r") as f:
        artists = json.load(f)

    all_tracks = []
    track_names = set()

    for artist in artists:
        artist_tracks = sp.artist_top_tracks(artist["uri"])["tracks"]
        artist_tracks = [t for t in artist_tracks if t["name"] not in track_names]

        all_tracks += artist_tracks
        track_names.update(t["name"] for t in artist_tracks)

    # In addition to all the raw track data, create a version
    # of the data which can easily be skimmed by a human
    simple_tracks = [(t["name"], t["artists"][0]["name"], t["popularity"])
                     for t in all_tracks]
    sorted_tracks = sorted(simple_tracks, reverse=True,
                           key=lambda track: track[2])

    return all_tracks, sorted_tracks


if __name__ == "__main__":
    for genre in ("kpop", "pop"):
        artists_file = os.path.join("data", f"{genre}.json")
        tracks, simple_tracks = get_popular_tracks(artists_file)

        utils.write_json(tracks, os.path.join("data", f"{genre}-tracks.json"))
        utils.write_json(simple_tracks, os.path.join("data", f"{genre}-tracks-simple.json"))