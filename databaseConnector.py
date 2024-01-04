import mysql.connector
from databaseConfig import connect
import csv

db = connect()

cursor = cursor = db.cursor()


def addTracks(tracks):
    for track in tracks:
        batchId = track['batchId']
        sql = "SELECT track_id FROM tracks WHERE track_id = %s"
        values = (track['id'], )
        cursor.execute(sql, values)
        existing = cursor.fetchall()

        if not existing:
            sql = "INSERT INTO features (featureset_id, danceability, energy, `key`, loudness, mode, speechiness, acousticness, instrumentalness, liveleness, valence, tempo, duration) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            values = (track['id'], track['features']['danceability'], track['features']['energy'], track['features']['key'],
                      track['features']['loudness'], track['features']['mode'], track['features']['speechiness'], track['features']['acousticness'],
                      track['features']['instrumentalness'], track['features']['liveness'], track['features']['valence'], track['features']['tempo'], track['features']['duration_ms'])
            cursor.execute(sql, values)

            sql = "INSERT INTO tracks (track_id,track_name,preview_url,spotify_url,featureset_id,time_added,trained,batch) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
            values = (track['id'], track['name'], track['preview_url'], track['spotify_url'],
                      track['id'], track['timestamp'], False, batchId)
            cursor.execute(sql, values)
            print("Added " + str(track['name'] + " to track table"))

            sql = "INSERT INTO artists_tracks VALUES (%s,%s)"
            values = (artist["artist_id"], track['id'])
            cursor.execute(sql, values)

            sql = "INSERT INTO track_genres (track_id, genre_id) VALUES (%s, %s)"
            values = (track['id'], rowid[0])
            cursor.execute(sql, values)

            sql = "INSERT INTO album_tracks (track_id, album_id) VALUES (%s, %s)"
            values = (track['id'], track['album']['id'])
            cursor.execute(sql, values)

        db.commit()


def addArtist(id, name, url):
    sql = "SELECT * FROM artists WHERE artist_id = %s"
    values = (id, )
    cursor.execute(sql, values)
    existing = cursor.fetchall()
    if not existing:
        sql = "INSERT INTO artists VALUES (%s,%s,%s)"
        values = (id, name, url)
        cursor.execute(sql, values)
        print("Added " + str(name + " to artist table"))

def populateGenreTable():
    with open('fma_data/genres.csv', 'r',encoding='utf-8') as genres:
        reader = csv.reader(genres)
        next(reader)
        for i, genre in enumerate(reader, start=1):
            cursor.execute("SELECT * FROM genres WHERE genre_id = %s", (genre[0],))
            exists = cursor.fetchone()
            if exists:
                print("This genre ID exists.")
            else:
                sql = "INSERT INTO genres (genre_id, genre_parent, top_genre, genre_name) VALUES (%s,%s,%s,%s)"
                values = (genre[0],genre[2],genre[4],genre[3])
                cursor.execute(sql, values)
    db.commit()

def populateAlbumTable():
    with open('fma_data/raw_albums.csv', 'r',encoding='utf-8') as albums:
        reader = csv.reader(albums)
        next(reader)
        for i, album in enumerate(reader, start=1):
            cursor.execute("SELECT * FROM albums WHERE album_id = %s", (album[0],))
            exists = cursor.fetchone()
            if exists:
                print("This album ID exists.")
            else:
                sql = "INSERT INTO albums (album_id, album_link, album_name) VALUES (%s,%s,%s)"
                values = (album[0], album[15], album[6])
                cursor.execute(sql, values)
                cursor.execute("SELECT artist_id FROM artists WHERE artist_name = %s", (album[16],))
                exists = cursor.fetchone()
                if exists:
                    sql = "INSERT INTO artists_albums (artist_id, album_id) VALUES (%s,%s)"
                    values = (exists[0], album[0])
                    cursor.execute(sql, values)
    db.commit()

def populateArtistTable():
    with open('fma_data/raw_artists.csv', 'r',encoding='utf-8') as artists:
        reader = csv.reader(artists)
        next(reader)
        for i, artist in enumerate(reader, start=1):
            cursor.execute("SELECT * FROM artists WHERE artist_id = %s", (artist[0],))
            exists = cursor.fetchone()
            if exists:
                print("This artist ID exists.")
            else:
                sql = "INSERT INTO artists (artist_id, artist_name, artist_url) VALUES (%s,%s,%s)"
                values = (artist[0], artist[11], artist[21])
                cursor.execute(sql, values)
    db.commit()