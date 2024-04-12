from flask import Flask, request, redirect, url_for, render_template, session, after_this_request,flash,jsonify
from werkzeug.utils import secure_filename
import torch
from genreModel import topGenreClassifier
from featureModel import audioFeatureModel
from subGenreModel import SubGenreClassifier
import os
from pydub import AudioSegment
import io
from featureExtraction import gen_spectrogram_path
from featureExtraction import gen_mffc_path
from featureExtraction import get_rhythm_info_path
from featureExtraction import plot_spectrogram
from prepareDataset import all_top_genre_names
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
import io
import base64
import json
from uuid import UUID
from collections import Counter
from annoy import AnnoyIndex
from itertools import islice
from databaseConfig import connect
import csv
from werkzeug.utils import secure_filename
from flask import send_file
import shutil
from maxUtils import initialize_sub_genre_models, get_top_to_sub_genre_map
from collections import Counter, defaultdict, OrderedDict

app = Flask(__name__)
app.secret_key = os.urandom(24)

script_path = os.path.abspath(__file__)
base_path = os.path.dirname(script_path)

genre_model = topGenreClassifier()
genre_model.load_state_dict(torch.load('max_genre_v5.pth'))
genre_model.eval()

feature_model = audioFeatureModel()
feature_model.load_state_dict(torch.load('max_feature_v4.pth'))
feature_model.eval()


sub_genre_models = initialize_sub_genre_models()
""" sub_genre_path = os.path.join("static","data","models","sub_models")
for file in os.listdir(sub_genre_path):
    filename = os.fsdecode(file)
    if filename.endswith(".pth"):
        id = int(filename.split('_')[3].split('.')[0])
        model = sub_genre_models[id]
        model.load_state_dict(torch.load(os.path.join(sub_genre_path,file)))
        model.eval()
        sub_genre_models[id] = model
    else:
        continue """

genre_features = []
with open("genre_averages.csv", 'r') as file:
    reader = csv.reader(file)
    for i, genre in enumerate(reader):
        genre_features.append(genre[1:9])

app.config['UPLOAD_FOLDER'] = os.path.join(base_path, 'upload')
app.config['TOP_GENRES'] = all_top_genre_names()
app.config['GENRE_MAP'] = get_top_to_sub_genre_map()
app.static_folder = 'static'

db = connect()
cursor = db.cursor()

@app.route('/')
def index():
    """ Serve the main index page. """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle file uploads through POST requests.

    Processes the uploaded file, extracts features, and predicts genre using the trained models.

    Returns:
    JSON response with status and redirect on success.
    """
    if 'file' not in request.files:
        flash('No file part', 'error')
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if file: 
        folder_path = extract_audio(file)
        if folder_path == None:
            return jsonify({'status': 'error', 'message': 'File too short, must be over 30 seconds'}), 400
        predict(folder_path)
        session['filename'] = secure_filename(file.filename)
        return redirect(url_for('display_analysis'))
    
@app.route('/results')
def display_analysis():
    """
    Display the results of the audio analysis.

    Fetches and sorts data from the session, then renders the results on a web page.
    """
    genre_info = session.get('genres', [])
    if not genre_info:
        return redirect(url_for('index'))
    
    genre_info = OrderedDict(sorted(genre_info.items(), key=lambda x: (-x[1]['probability'], x[0])))
    
    feature_averages = session['features']
    file_id = session['fileId']
    beat_grid = session['beat_grid']

    chart_data = [feature_averages['Acousticness'], feature_averages['Danceability'], feature_averages['Energy'], feature_averages['Instrumentalness'], feature_averages['Liveness'], feature_averages['Speechiness'], feature_averages['Valence']]

    nearest_track_info = get_nearest_tracks(feature_averages)

    return render_template('analysis_results.html', file_id = file_id, 
                           genres = genre_info, features = feature_averages,filename = session['filename'], nearest_track_info = nearest_track_info, 
                           chart_data = chart_data, average_genre_features = genre_features, beat_grid = beat_grid)

@app.route('/stream-audio')
def stream_audio():
    """
    Stream the uploaded audio file.

    Returns:
    - Audio file or error message.
    """
    temp_path = session.get('file_path')
    if temp_path:
        return send_file(temp_path, as_attachment=False)
    return "No audio file available", 404

def predict(folder_path):
    """
    Predict the genre and sub-genre of the uploaded audio files.

    Parameters:
    - folder_path (str): Path where the uploaded audio files are stored.

    Returns:
    - list: A list of prediction results.
    """
    clear_session()
    sigmoid = torch.nn.Sigmoid()

    files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]

    full_prediction_data = []

    for idx, file in enumerate(files):
        file_path = os.path.join(folder_path,file)
        session['file_path'] = file_path
        spec_plot, spec, file_id = gen_spectrogram_path(file_path)    
        mfcc = gen_mffc_path(file_path)
        beats = get_rhythm_info_path(file_path)  
        if idx == 0:
            plot_spectrogram(file_id, spec_plot)

        spec = spec[..., :2580]
        mfcc = mfcc[..., :2580]
        beat_vector = torch.zeros(mfcc.shape[2])
        beat_vector[beats[1]] = 1
        beat_grid = [i for i, value in enumerate(beat_vector.to(torch.int).tolist()) if value == 1]
        session['beat_grid'] = beat_grid
        beat_vector = beat_vector.unsqueeze(0).unsqueeze(0)
        beat_vector = beat_vector.repeat(2,1,1)

        combined_features = torch.cat([spec, mfcc, beat_vector], dim=1).unsqueeze(0)
        with torch.no_grad():
            genre_prediction = genre_model(combined_features)
            feature_prediction = feature_model(combined_features)

            genre_probabilities = sigmoid(genre_prediction.data)

            top3_values, top3_indices = torch.topk(genre_probabilities, k=3, dim=1) 
            top3_probabilities = top3_values.cpu().numpy()
            top3_predictions = top3_indices.cpu().numpy()

            top_values, top_indices = torch.topk(genre_probabilities, k=1, dim=1)
            top_prediction = top_indices.cpu().numpy()
            top_probability = top_values.cpu().numpy()
            
            sub_genre_results = {}
            for genre in top3_indices.tolist()[0]:
                sub_genre_info = {}
                sub_genre_model = sub_genre_models[genre]
                sub_genre_prediction = sub_genre_model(combined_features)
                sub_genre_probabilities = sigmoid(sub_genre_prediction.data)
                if not torch.isnan(sub_genre_probabilities).all():
                    if len(sub_genre_probabilities[0]) < 3:
                        k = len(sub_genre_probabilities[0])
                    else:
                        k = 3
                    top3_sub_genre_values, top3_sub_genre_indices = torch.topk(sub_genre_probabilities, k=k, dim=1)
                    sub_genre_info['top3_probabilities'] = top3_sub_genre_values.cpu().numpy()
                    sub_genre_info['top3_genres'] = top3_sub_genre_indices.cpu().numpy()
                    sub_genre_results[genre] = sub_genre_info

        top3_genres = {}
        top_genres = app.config['TOP_GENRES']
        for idx, genre_prediction in enumerate(top3_predictions[0]):
            if idx not in top3_genres:
                top3_genres[idx] = {}
            top3_genres[idx]['top_genre'] = [top_genres[genre_prediction], int(top3_values[0][idx]*100)]
            if genre_prediction in sub_genre_results:
                sub_genre_ids = []
                for index in sub_genre_results[genre_prediction]['top3_genres'].tolist()[0]:
                    sub_genre_ids.append(get_sub_genre_id(top_genres[genre_prediction], index))
                sub_genre_results[genre_prediction]['top3_genres'] = sub_genre_ids
                top3_genres[idx]['sub_genres'] = sub_genre_results[genre_prediction]

        top_prediction = top_genres[top_prediction[0][0]]

        feature_prediction = feature_prediction.tolist()[0]

        features = {
            "Danceability": feature_prediction[0],
            "Energy": feature_prediction[1],
            "Speechiness": feature_prediction[2],
            "Acousticness": feature_prediction[3],
            "Instrumentalness": feature_prediction[4],
            "Liveness": feature_prediction[5],
            "Valence": feature_prediction[6],
            "tempo": beats[0].item()
        }
    
        prediction_data = {
            "file_id": file_id,
            "top_prediction": {
              "genre": top_prediction,
              "probability": top_probability
            },
            "top_3_predictions": top3_genres,
            "feature_predictions": features
        }
        full_prediction_data.append(prediction_data)

    combine_predictions(full_prediction_data)

    return full_prediction_data

def extract_audio(file):
    """
    Extracts and splits an audio file into multiple segments.

    Parameters:
    - file (werkzeug.datastructures.FileStorage): The uploaded file object.

    Returns:
    - str: Path to the folder containing extracted audio clips if successful, None otherwise.
    """
    split_size = 30000
    audio = AudioSegment.from_file(file)
    if len(audio) < 31000:
        return None
    base_name = file.filename
    output_folder = os.path.join(os.getcwd(),"upload", os.path.splitext(base_name)[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        full_clips_count = len(audio) // split_size
        for i in range(full_clips_count):
            start = i * split_size
            end = start + split_size
            clip = audio[start:end]
            clip.export(os.path.join("upload",output_folder, f"clip_{i+1}.mp3"), format="mp3")
    return output_folder

def serialize_dict(data):
    """
    Recursively converts data structures into serializable formats.

    Parameters:
    - data (dict, list, np.ndarray, int, float, str, bool): Input data to be serialized.

    Returns:
    - Various (dict, list, int, float, str): Serialized data ready for JSON or other types of output.
    """
    if isinstance(data, dict):
        return {key: serialize_dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_dict(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (int, float, str, bool)):
        return data
    else:
        return str(data)
    
def format_time(seconds):
    """
    Formats seconds into a minute:second string.

    Parameters:
    - seconds (int): Number of seconds.

    Returns:
    - str: Formatted time string.
    """
    return f"{seconds//60}:{seconds%60:02d}"

def get_nearest_tracks(track_features):
    """
    Finds and retrieves nearest tracks from an ANN index based on input features.

    Parameters:
    - track_features (dict): Features of a track used to find nearest neighbors.

    Returns:
    - list: List of track information including name, artist, and similarity distance.
    """
    ann_index = AnnoyIndex(6, 'euclidean')
    ann_index.load('static/data/ANN_tracks_index.ann')
    track_ids, distances = ann_index.get_nns_by_vector(list(islice(track_features.values(), 6)), 9, include_distances=True)
    del ann_index

    track_info = []
    for idx, track in enumerate(track_ids):
        query = """
        SELECT 
            t.track_name,
            t.explicit,
            ar.artist_name, 
            ar.artist_url AS artist_link,
            alb.album_name,  
            alb.album_link AS album_link
        FROM tracks t
        JOIN album_tracks at ON t.track_id = at.track_id
        JOIN albums alb ON at.album_id = alb.album_id
        JOIN artists_tracks art ON t.track_id = art.track_id
        JOIN artists ar ON art.artist_id = ar.artist_id
        WHERE t.track_id = %s;
        """
        # TODO something weird here with tracks
        cursor.execute(query,(track,))
        track = cursor.fetchone()
        track = list(track)
        track.append(distances[idx])
        track_info.append(track)
    return track_info

def average_dict(dict):
    """
    Averages numerical values stored in dictionaries.

    Parameters:
    - d (dict): Dictionary whose values are lists of numbers.

    Returns:
    - dict: Dictionary with the same keys and average values.
    """
    averages = {}
    for item, values in dict.items():
        if isinstance(values, list):
            average_value = sum(values) / len(values)
            averages[item] = round(average_value,2)
    return averages

def get_sub_genre_id(top_genre_id, sub_genre_idx):
    """
    Fetches the sub-genre ID based on its index and parent genre ID.

    Parameters:
    - top_genre_id (int): ID of the top genre.
    - sub_genre_idx (int): Index of the sub-genre.

    Returns:
    - int: Sub-genre ID.
    """
    top_genre_id = top_genre_id[1]
    query = """
    SELECT genre_id FROM genres
    WHERE top_genre = %s
    ORDER BY genre_id;
    """
    values = (int(top_genre_id),)
    cursor.execute(query, values)
    sub_genre_ids = [row[0] for row in cursor.fetchall()]
    return sub_genre_ids[sub_genre_idx]

def get_genre_name(genre_id):
    query = """
    SELECT genre_name FROM genres
    WHERE genre_id = %s
    """
    values = (int(genre_id),)
    cursor.execute(query, values)
    return cursor.fetchone()[0]

def combine_predictions(prediction_list):
    """
    Aggregates and processes prediction results from multiple analyzed files.

    Parameters:
    - prediction_list (list): A list of dictionaries containing predictions for each processed file.

    Effects:
    - Processes and aggregates genre and feature predictions across multiple files.
    - Stores combined results in the session for later use in the session context.

    Stores:
    - Averages of features and genre probabilities in the session.
    - Detailed genre and sub-genre predictions including the probability distributions.
    """
    features = {}
    genre_probabilities = defaultdict(list)
    genre_counts = Counter()
    genre_ids = defaultdict(int)
    sub_genre_counts = Counter()
    sub_genre_probabilities = defaultdict(list)
    
    for file in prediction_list:
        for prediction in file['top_3_predictions']:
            top_genre_entry = file['top_3_predictions'][prediction]['top_genre']
            genre, probability = top_genre_entry
            genre_counts[genre] += 1
            genre_probabilities[genre].append(probability)
            genre_ids[genre] = genre
            
            sub_genres_info = file['top_3_predictions'][prediction]['sub_genres']
            for sub_genre_id, prob in zip(sub_genres_info['top3_genres'], sub_genres_info['top3_probabilities'][0]):
                sub_genre_counts[sub_genre_id] += 1
                sub_genre_probabilities[sub_genre_id].append(prob)

        for feature, value in file["feature_predictions"].items():
            features.setdefault(feature, []).append(value)

    feature_averages = average_dict(features)
    genre_averages = average_dict(genre_probabilities)
    genre_averages = dict(sorted(genre_averages.items(), key=lambda item: item[1], reverse=True)[:5])

    genre_info = {genre: {'probability': prob, 'sub_genres': {}} for genre, prob in genre_averages.items()}

    for sub_genre_id, probs in sub_genre_probabilities.items():
        top_genre_id = app.config['GENRE_MAP'].get(sub_genre_id)
        if top_genre_id and genre_ids[top_genre_id] in genre_info:
            avg_prob = sum(probs) / len(probs) if probs else 0
            genre_name = get_genre_name(sub_genre_id)
            genre_info[genre_ids[top_genre_id]]['sub_genres'][genre_name] = avg_prob

    session['genres'] = genre_info
    session['features'] = feature_averages
    session['fileId'] = prediction_list[0]['file_id']
    
    plot_feature_over_time(features, prediction_list[0]['file_id'])

def plot_feature_over_time(features, file_id):
    """
    Plots and saves a graph of music features over time.

    Parameters:
    - features (dict): Dictionary of features extracted from audio analysis.
    - file_id (str): Unique identifier for the current file/session used to save the plot.
    """
    time = np.arange(0, len(next(iter(features.values()))) * 30, 30)
    formatted_time = [format_time(t) for t in time]
    plt.figure(figsize=(14, 10))
    for feature, values in features.items():
        if feature != "tempo":
            plt.plot(formatted_time, values, label=feature)
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Feature Value')
    plt.title('Music Features Over Time')
    plt.legend()
    plt.savefig(f"static/images/{file_id}-features.png", format='png', bbox_inches='tight')
    plt.close()

def clear_session():
    """
    Clears all session variables related to the current analysis session.
    """
    if 'beat_grid' in session:
        del session['beat_grid']
    if 'genres' in session:
        del session['genres']
    if 'features' in session:
        del session['features']
    if 'fileId' in session:
        del session['fileId']
    if 'filename' in session:
        del session['filename']
    if 'file_path' in session:
        del session['file_path']

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.run(debug=True)
