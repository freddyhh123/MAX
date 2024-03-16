from flask import Flask, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
import torch
from genreModel import topGenreClassifier
from featureModel import audioFeatureModel
import os
from pydub import AudioSegment
import io
from featureExtraction import gen_spectrogram_path
from featureExtraction import gen_mffc_path
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

app = Flask(__name__)
app.secret_key = os.urandom(24)

script_path = os.path.abspath(__file__)
base_path = os.path.dirname(script_path)

genre_model = topGenreClassifier()
genre_model.load_state_dict(torch.load('max_genre_v1.pth'))
genre_model.eval()

feature_model = audioFeatureModel()
feature_model.load_state_dict(torch.load('max_feature_v1.pth'))
feature_model.eval()

app.config['UPLOAD_FOLDER'] = os.path.join(base_path, 'upload')
app.config['TOP_GENRES'] = all_top_genre_names()
app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        folder_path = extract_audio(file)
        predictions = predict(folder_path)
        session['predictions'] = serialize_dict(predictions)

        return redirect(url_for('display_analysis'))
    
@app.route('/results')
def display_analysis():
    prediction_list = session.get('predictions', [])
    if not prediction_list:
        return redirect(url_for('index'))

    genre_probabilities = {}
    genre_counts = Counter()
    features = {}

    for file in prediction_list:
        for prediction_str in file['top_3_predictions']:
            prediction_str = re.sub(r"tensor\(([^)]+)\)", r"\1", prediction_str)
            (genre, _), probability = ast.literal_eval(prediction_str)
            genre_counts[genre] += 1
            if genre in genre_probabilities:
                genre_probabilities[genre].append(probability)
            else:
                genre_probabilities[genre] = [probability]

        for feature, value in file["feature_predictions"].items():
            if feature in features:
                features[feature].append(value)
            else:
                features[feature] = [value]


    most_common_genres = [genre for genre, _ in genre_counts.most_common(3)]

    plot_genres = []
    plot_probabilities = []
    for genre in most_common_genres:
        average_probability = sum(genre_probabilities[genre]) / len(genre_probabilities[genre])
        plot_genres.append(genre)
        plot_probabilities.append(average_probability)

    feature_averages = {}
    for feature, values in features.items():
        average_value = sum(values) / len(values)
        feature_averages[feature] = round(average_value,2)

    genre_averages = {}
    for genre, values in genre_probabilities.items():
        average_value = sum(values) / len(values)
        genre_averages[genre] = round(average_value,2)

    genre_buf = io.BytesIO()
    plt.figure(figsize=(10, 6),facecolor='none', edgecolor='none')
    plt.bar(plot_genres, plot_probabilities, color='purple')
    plt.xlabel('Genres', color='white')
    plt.ylabel('Average Probability', color='white')
    plt.title('Top 3 Most Common Genres and Their Average Probabilities across the track', color='white')
    plt.xticks(rotation=45,color='white')
    plt.yticks(color='white')
    plt.gca().set_facecolor('none')
    plt.savefig(genre_buf, format='png', bbox_inches='tight')
    genre_buf.seek(0)
    genre_plot_url = base64.b64encode(genre_buf.getvalue()).decode('utf-8')
    genre_buf.close()

    tempo = features.popitem()

    feature_buf = io.BytesIO()
    time = np.arange(0, len(next(iter(features.values()))) * 30, 30)
    time = [format_time(t) for t in time]
    plt.figure(figsize=(14, 10))
    for feature, values in features.items():
        plt.plot(time, values, label = feature)
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Feature Value')
    plt.title('Music Features Over Time')
    plt.legend()
    plt.savefig(feature_buf, format='png', bbox_inches='tight')
    feature_plot_url = base64.b64encode(feature_buf.getvalue()).decode('utf-8')
    feature_buf.close()
    
    return render_template('analysis_results.html',feature_plot_url = feature_plot_url, genre_plot_url=genre_plot_url, file_id = prediction_list[0]['file_id'], genres = genre_averages, features = feature_averages)

def predict(folder_path):
    sigmoid = torch.nn.Sigmoid()

    files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]

    full_prediction_data = []

    for file in files:
        file_path = os.path.join(folder_path,file)
        spec, file_id = gen_spectrogram_path(file_path)
        mfcc = gen_mffc_path(file_path)
        spec = spec[..., :2580]
        mfcc = mfcc[..., :2580]

        combined_features = torch.cat([spec, mfcc], dim=1)
        with torch.no_grad():
            genre_prediction = genre_model(combined_features)
            feature_prediction = feature_model(combined_features.unsqueeze(0))

            genre_probabilities = sigmoid(genre_prediction.data)

            top3_values, top3_indices = torch.topk(genre_probabilities, k=3, dim=1) 
            top3_probabilities = top3_values.cpu().numpy()
            top3_predictions = top3_indices.cpu().numpy()

            top_values, top_indices = torch.topk(genre_probabilities, k=1, dim=1)
            top_prediction = top_indices.cpu().numpy()
            top_probability = top_values.cpu().numpy()
        
        top3_genres = list()
        top_genres = app.config['TOP_GENRES']
        for idx, genre_prediction in enumerate(top3_predictions[0]):
            top3_genres.append((top_genres[genre_prediction], top3_values[0][idx]*100))

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
            "tempo": feature_prediction[7]
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

    return full_prediction_data

def extract_audio(file):
    split_size = 30000
    audio = AudioSegment.from_file(file)
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
    return f"{seconds//60}:{seconds%60:02d}"

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.run(debug=True)
