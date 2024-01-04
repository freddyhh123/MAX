import processTracks
import databaseConnector
#

databaseConnector.populateArtistTable()
databaseConnector.populateAlbumTable()
databaseConnector.populateGenreTable()

tracks = processTracks.getTracks(None)
databaseConnector.addTracks(tracks)
1==1