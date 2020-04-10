import os
import json
import spotipy
import utils
from time import sleep
from spotipy.oauth2 import SpotifyClientCredentials


sp = utils.get_spotipy_instance()


def get_popular_tracks(artists_file: str):
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


def get_kpop_tracks(artists_file: str):
    with open(artists_file, "r") as f:
        artists = json.load(f)
    
    tracks = []

    for artist in artists:
        print(artist["id"])
        albums = sp.artist_albums(artist["id"], "album,single")
        
        for album in albums["items"]:
            tracks += sp.album_tracks(album["id"])["items"]

        sleep(1.0)

    return tracks


def output_track_features(tracks, output_path, throttle=False):
    all_tracks = []

    for track_batch in utils.batches(tracks, 50):
        track_ids = [track["id"] for track in track_batch]
        track_features = sp.audio_features(track_ids)
        all_tracks += track_features

        if throttle:
            sleep(1.0)

    # Originally saved track features in one file per
    # track, but found that it took too long for Spark
    # to go through all the files and parse them into
    # a DataFrame. Storing all track features in a single
    # file gives tremendously better performance (on a single machine).
    utils.write_json(all_tracks, output_path)


def general_popular_tracks():
    # Get popular tracks for each major genre
    for genre in ("kpop", "pop"):
        artists_file = os.path.join("data", f"{genre}.json")
        tracks, simple_tracks = get_popular_tracks(artists_file)

        utils.write_json(tracks, os.path.join("data", f"{genre}-tracks.json"))
        utils.write_json(simple_tracks, os.path.join("data", f"{genre}-tracks-simple.json"))

        # Get the audio features for the tracks we collected
        output_path = os.path.join("data", f"{genre}-track-features.json")
        output_track_features(tracks, output_path)


def kpop_detailed_tracks():
    tracks = []

    # Get *all* tracks for each of these artists
    for group_type in ("kpop-bg", "kpop-gg"):
        artists_file = os.path.join("data", f"{group_type}.json")
        tracks += get_kpop_tracks(artists_file)
        
    utils.write_json(tracks, os.path.join("data", f"kpop-tracks-lg.json"))

    # Get the audio features for the tracks we collected
    # This is a lot more tracks, so adding some rest between
    # requests to avoid getting rate-limited
    output_path = os.path.join("data", f"kpop-track-features-lg.json")
    output_track_features(tracks, output_path, throttle=True)


if __name__ == "__main__":
    # general_popular_tracks()
    kpop_detailed_tracks()