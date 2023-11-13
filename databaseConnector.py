import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="max"
)

cursor = cursor = db.cursor()


def addTracks(tracks):
    for track in tracks:
        batchId = track['batchId']
        sql = "SELECT track_id FROM tracks WHERE track_id LIKE '" + \
            str(track['id']) + "'"
        cursor.execute(sql)
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

            for artist in track['artists']:
                addArtist(artist['artist_id'],
                          artist['artist_name'], artist['artist_url'])
                sql = "INSERT INTO artists_tracks VALUES (%s,%s)"
                values = (artist["artist_id"], track['id'])
                cursor.execute(sql, values)

            for genre in track['genres']:
                sql = "SELECT * FROM genres WHERE genre_name LIKE '" + genre + "'"
                cursor.execute(sql)
                existing = cursor.fetchall()
                if not existing:
                    sql = "INSERT INTO genres (genre_name) VALUES ('" + \
                        genre + "')"
                    cursor.execute(sql)
                    print("Added " + genre + " to genres table")
                    db.commit()
                rowid = "SELECT genre_id FROM genres WHERE genre_name LIKE '"+ genre +"'"
                cursor.execute(rowid)
                rowid = cursor.fetchone()
                sql = "INSERT INTO track_genres (track_id,genre_id) VALUES ('" + \
                    track['id']+"',"+str(rowid[0])+")"
                cursor.execute(sql)

            sql = "SELECT album_id FROM albums WHERE album_id LIKE '" + \
                str(track['album']['id']) + "'"
            cursor.execute(sql)
            existing = cursor.fetchall()

            if not existing:
                sql = "INSERT INTO albums (album_id,album_name,spotify_url) VALUES (%s,%s,%s)"
                values = (track['album']['id'], track['album']['name'], track['album']['url'])
                cursor.execute(sql, values)

                for image in track['album']['images']:
                    sql = "INSERT INTO album_images (image_url,image_height,image_width) VALUES (%s,%s,%s)"
                    values = (image['url'], image['height'], image['width'])
                    cursor.execute(sql, values)
                    db.commit()
                    rowid = "SELECT image_id FROM album_images WHERE image_url LIKE '"+ image['url']+"'"
                    cursor.execute(rowid)
                    rowid = cursor.fetchone()
                    sql = "INSERT INTO image_links (album_id,image_id) VALUES ('" + \
                        track['album']['id']+"',"+str(rowid[0])+")"
                    cursor.execute(sql)

                for artist in track['album']['artists']:
                    addArtist(artist['artist_id'], artist['artist_name'], artist['artist_url'])
                    sql = "INSERT INTO artists_albums (artist_id,album_id) VALUES ('" + \
                        artist['artist_id']+"','"+track['album']['id']+"')"
                    cursor.execute(sql)

            sql = "INSERT INTO album_tracks (track_id,album_id) VALUES ('" + \
                track['id']+"','"+track['album']['id']+"')"
            cursor.execute(sql)

        db.commit()


def addArtist(id, name, url):
    sql = "SELECT * FROM artists WHERE artist_id ='" + str(id) + "'"
    cursor.execute(sql)
    existing = cursor.fetchall()
    if not existing:
        sql = "INSERT INTO artists VALUES (%s,%s,%s)"
        values = (id, name, url)
        cursor.execute(sql, values)
        print("Added " + str(name + " to artist table"))
