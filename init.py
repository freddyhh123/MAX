import processTracks
import databaseConnector
import prepareDataset
import fmaDataset
import torch.optim as optim
#

#databaseConnector.populateArtistTable()
#databaseConnector.populateAlbumTable()
#databaseConnector.populateGenreTable()

#tracks = processTracks.getTracks(None)
#databaseConnector.addTracks(tracks)

track_dataframe = prepareDataset.buildDataframe()

model = fmaDataset(dataframe = track_dataframe, spectrogram = track_dataframe['spectrogram'].values, centroid = track_dataframe['centroid'], labels = track_dataframe['genre_vector'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
1==1