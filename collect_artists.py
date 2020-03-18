import os
import errno
import sys
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

SEARCH_LIMIT = 50
GENRES       = { "k-pop+boy+group" : 42, 
                 "k-pop+girl+group": 33 }
GENRE_FILES  = { "k-pop+boy+group" : "data/bg.json", 
                 "k-pop+girl+group": "data/gg.json" }

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

for genre, cutoff in GENRES.items():
    search_results = sp.search(
        q=f"genre:{genre}",
        type="artist",
        limit=50)

    artists = search_results["artists"]["items"]
    artists = sorted(artists,
                    reverse=True, 
                    key=lambda artist: artist["popularity"])
    artists = [artist["name"] for artist in artists[:cutoff]]
    output_path = GENRE_FILES[genre]

    try:
        os.makedirs(os.path.dirname(output_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(output_path, "w") as f:
        json.dump(artists, f)