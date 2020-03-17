from pprint import pprint
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

artists = []
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

for index in range(3):
    search_results = sp.search(
        q="genre:k-pop", 
        type="artist", 
        limit=50, 
        offset=index * 50)
    artists += search_results["artists"]["items"]

artists = sorted(artists, 
                 reverse=True, 
                 key=lambda artist: artist["popularity"])

for index, artist in enumerate(artists):
    print(index, artist["name"])