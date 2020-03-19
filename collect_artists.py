import os
import errno
import sys
import json
import spotipy
import utils
from spotipy.oauth2 import SpotifyClientCredentials


SEARCH_LIMIT = 50
GENRES       = { "k-pop"           : 50,
                 "k-pop+boy+group" : 42, 
                 "k-pop+girl+group": 33,
                 "pop"             : 50 }
GENRE_FILES  = { "k-pop"           : "data/kpop.json",
                 "k-pop+boy+group" : "data/kpop-bg.json", 
                 "k-pop+girl+group": "data/kpop-gg.json",
                 "pop"             : "data/pop.json" }


sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
artists_by_genre = {}


for genre, cutoff in GENRES.items():
    search_results = sp.search(
        q=f"genre:{genre}",
        type="artist",
        limit=50)

    artists = search_results["artists"]["items"]
    artists = sorted(artists,
                    reverse=True, 
                    key=lambda artist: artist["popularity"])
    artists = artists[:cutoff]
    artists_by_genre[genre] = set(artist["name"] for artist in artists)

    # Some k-pop groups show up under the pop category as well
    if genre == "pop":
        artists = [artist for artist in artists 
                   if artist["name"] not in artists_by_genre["k-pop"]]

    utils.write_json(artists, GENRE_FILES[genre])