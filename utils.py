import os
import errno
import json
import math
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def euclid_dist(x, y):
    sum_of_squares = 0

    for xi, yi in zip(x, y):
        sum_of_squares += ((xi - yi) ** 2)

    return math.sqrt(sum_of_squares)


def batches(iterable, batch_size):
    num_batches = math.ceil(len(iterable) / batch_size)
    for i in range(num_batches):
        offset = i * batch_size
        yield iterable[offset:offset + batch_size]


def get_spotipy_instance():
    return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


def makedirs(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def write_json(result, output_path):
    makedirs(output_path)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
