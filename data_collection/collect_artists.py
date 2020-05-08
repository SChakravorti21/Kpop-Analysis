import os
import errno
import sys
import json
import math
import utils

SEARCH_LIMIT = 50
GENRE_FILES  = { "k-pop"    : "data/kpop.json",
                 "k-pop-lg" : "data/kpop-lg.json", 
                 "pop"      : "data/pop.json" }


def collect_artists(genre, num_artists=SEARCH_LIMIT):
    sp = utils.get_spotipy_instance()
    artists = []

    for batch in utils.batches(range(num_artists), SEARCH_LIMIT):
        search_results = sp.search(
            q=f"genre:{genre}",
            type="artist",
            offset=batch[0],
            limit=len(batch)
        )

        artists += search_results["artists"]["items"]
    
    return sorted(
        artists,
        reverse=True, 
        key=lambda artist: artist["popularity"]
    )


def collect_pop_kpop_artists():
    # Collect top 50 K-pop artists
    kpop_artists = collect_artists("k-pop")
    kpop_artist_names = set(a["name"] for a in kpop_artists)
    
    # Collect top 50 pop artists, filter out
    # any artists who also showed up under K-pop 
    pop_artists = [a for a in collect_artists("pop") 
                   if a["name"] not in kpop_artist_names]

    utils.write_json(pop_artists, GENRE_FILES["pop"])
    utils.write_json(kpop_artists, GENRE_FILES["k-pop"])


def collect_many_kpop_artists():
    artists = collect_artists("k-pop", num_artists=150)
    utils.write_json(artists, GENRE_FILES["k-pop-lg"])


if __name__ == "__main__":
    collect_pop_kpop_artists()
    collect_many_kpop_artists()