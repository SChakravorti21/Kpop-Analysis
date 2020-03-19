import os
import errno
import json
import math
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def batches(iterable, batch_size):
    num_batches = math.ceil(len(iterable) / batch_size)
    for i in range(num_batches):
        offset = i * batch_size
        yield iterable[offset:offset + batch_size]


def get_spotipy_instance():
    return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


def write_json(result, output_path):
    try:
        os.makedirs(os.path.dirname(output_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)