import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import mysql.connector
from databaseConfig import connect
from spotipy_random import get_random

db = connect()

cursor = db.cursor()

cid = "516f5defb70e4e0994acea4cc99d1d4c"
secret = "0038b1fca65d467dbd0d2e1437041545"

client_credentials_manager = SpotifyClientCredentials(
client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10, retries=10)

genres = sp.recommendation_genre_seeds()['genres']
genres = [(item,) for item in genres]
active_genres = list()

for genre in genres:
    try:
        track = get_random(spotify=sp, type="track", genre = str(genre[0]))
        active_genres.append(genre)
    except:
        print('Skip genre:' + genre[0])
        continue

sql = "INSERT INTO seed_genres (genre_name) VALUES (%s)"

cursor.executemany(sql, active_genres)

db.commit()
cursor.close()
db.close()