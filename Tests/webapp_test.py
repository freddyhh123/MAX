import os
import unittest
import tempfile
import io
from max import app
from io import BytesIO
from pydub import AudioSegment
from max import extract_audio

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        app.config['DATABASE'] = self.db_path
        app.config['TESTING'] = True
        self.client = app.test_client()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)

    def test_upload(self):
        with open(os.path.join('Tests', 'audio', 'test.mp3'), 'rb') as f:
            data = {
                'file': (io.BytesIO(f.read()), 'test.mp3')
            }
            response = self.client.post('/upload', content_type='multipart/form-data', data=data)
            self.assertEqual(response.status_code, 302)

    def test_results(self):
        with self.client as c:
            with c.session_transaction() as sess:
                sess['genres'] = {'Electronic': {'probability': 37.5, 'sub_genres': {'Bigbeat': 0.5612871944904327, 'House': 0.561206042766571, 'Techno': 0.5640366673469543}}, 'Folk': {'probability': 38.0, 'sub_genres': {'British Folk': 0.49770618975162506, 'Folk': 0.49247273802757263, 'Free-Folk': 0.5054022669792175}}, 'Hip-Hop': {'probability': 20.0, 'sub_genres': {'Hip-Hop': 0.5421676635742188, 'Nerdcore': 0.5544494390487671, 'Wonky': 0.5281047821044922}}, 'Instrumental': {'probability': 17.0, 'sub_genres': {'Ambient': 0.5436880588531494, 'New Age': 0.5455314517021179, 'Soundtrack': 0.5767213702201843}}, 'International': {'probability': 21.0, 'sub_genres': {'Afrobeat': 0.5600001811981201, 'Pacific': 0.5599216818809509, 'Salsa': 0.5652175545692444}}}
                sess['features'] = {'Acousticness': 0.09, 'Danceability': 0.45, 'Energy': 0.66, 'Instrumentalness': 0.68, 'Liveness': 0.19, 'Speechiness': 0.09, 'Valence': 0.2, 'tempo': 161.5}
                sess['fileId'] = "83c79d0d-40d2-4500-99e1-19e4bb8f6738"
                sess['beat_grid'] = [3, 41, 74, 106, 138, 171, 203, 235, 267, 299, 332, 365, 397, 429, 461, 494, 526, 558, 591, 623, 655, 688, 719, 752, 784, 816, 849, 881, 913, 946, 978, 1011, 1042, 1075, 1108, 1140, 1173, 1204, 1236, 1269, 1301, 1333, 1366, 1398, 1431, 1462, 1494, 1527, 1560, 1592, 1625, 1657, 1690, 1721, 1753, 1785, 1818, 1851, 1884, 1916, 1948, 1980, 2012, 2044, 2077, 2109, 2141, 2174, 2207, 2238, 2271, 2303, 2335, 2368, 2402, 2434, 2466]
                sess['filename'] = "test.mp3"
        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)

    def test_stream_audio(self):
        path = str(os.path.join('Tests','audio','test.mp3'))
        with self.client as c:
            with c.session_transaction() as sess:
                sess['file_path'] = path
            response = c.get('/stream-audio')
        self.assertEqual(response.status_code, 200)
        self.assertIn('audio/mpeg', response.content_type)

    def test_short_audio_file(self):
            path = os.path.join('Tests', 'audio', 'small.mp3')
            with self.subTest("Check for handling of short audio"):
                folder_path = extract_audio(path)
                self.assertIsNone(folder_path, "The system should return None for an audio file shorter than 30 seconds")

if __name__ == '__main__':
    unittest.main()