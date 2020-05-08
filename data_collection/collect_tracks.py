import os
import json
import utils
from time import sleep


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

    # Tracks which songs have already been seen, because
    # very often popular songs are republished in newer albums,
    # but those would add unwanted bias to analyses
    visited_tracks = set()
    tracks = []

    for artist in artists:
        print(artist["id"])
        albums = sp.artist_albums(artist["id"], "album,single")

        for album in albums["items"]:
            for track in sp.album_tracks(album["id"])["items"]:
                track_artists = sorted([a["name"] for a in track["artists"]])
                track_key = (track["name"].lower(), *track_artists)

                if track_key not in visited_tracks \
                        and track["id"] not in visited_tracks:
                    # Add the album to the track info, not provided otherwise
                    track["album"] = album
                    visited_tracks.add(track_key)
                    visited_tracks.add(track["id"])
                    tracks.append(track)

        sleep(0.5)

    return tracks


def get_track_features(tracks, throttle=False):
    all_tracks = []

    for track_batch in utils.batches(tracks, 50):
        track_ids = [track["id"] for track in track_batch]
        print(track_ids[0])  # just print something to know stuff is happening
        track_features = sp.audio_features(track_ids)
        all_tracks += track_features

        if throttle:
            sleep(1.0)

    return all_tracks


def output_track_features(tracks, output_path, throttle=False):
    all_tracks = get_track_features(tracks, throttle=throttle)

    # Sometimes getting audio features might fail?
    # In any case, null values make it impossible to load
    # the dataset
    all_tracks = [track for track in all_tracks if track != None]

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
    # Get *all* tracks for each of these artists
    artists_file = os.path.join("data", f"kpop-lg.json")
    tracks = get_kpop_tracks(artists_file)
    utils.write_json(tracks, os.path.join("data", f"kpop-tracks-lg.json"))

    # Get the audio features for the tracks we collected
    # This is a lot more tracks, so adding some rest between
    # requests to avoid getting rate-limited
    output_path = os.path.join("data", f"kpop-track-features-lg.json")
    output_track_features(tracks, output_path, throttle=True)


if __name__ == "__main__":
    # general_popular_tracks()
    kpop_detailed_tracks()
