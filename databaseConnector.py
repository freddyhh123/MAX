import mysql.connector
import uuid

db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root"
)

cursor = mycursor = db.cursor()

def addTracks(tracks):
    batchId = uuid.uuid1()
    for track in tracks:
        1==1
        sql = "INSERT into tracks (track_id,track_name,preview_url,spotify_url,featureset_id,artist_id,album_id,time_added,trained,batch) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        values = (track['id'],track['name'],track['preview_url'],track['spotify_url'],track['id'],)