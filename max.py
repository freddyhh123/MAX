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
    return render_template('index.html')  # Render the front-end interface

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        file_path = extract_audio(file)
        predictions = predict(file_path)
        session['predictions'] = serialize_dict(predictions)

        return redirect(url_for('display_analysis'))
    
@app.route('/results')
def display_analysis():
    prediction_list = session.get('predictions', [])
    if not prediction_list:
        return redirect(url_for('index'))
    cleaned_predictions = re.sub(r'tensor\(([^)]+)\)', r'\1', prediction_list['top_3_predictions'])
    parsed_data = ast.literal_eval(cleaned_predictions)
    genre_dict = {}
    genres = list()
    probabilities = list()
    rounded_feature_list = prediction_list["feature_predictions"]
    for (genre, id), probability in parsed_data:
        genres.append(genre)
        probabilities.append(probability)
        genre_dict[genre] = (id, probability)
    for feature in prediction_list["feature_predictions"]:
        rounded_feature_list[feature] = round(rounded_feature_list[feature], 3)
    
    buf = io.BytesIO()
    plt.figure(figsize=(10, 6),facecolor='none', edgecolor='none')
    plt.bar(genres, probabilities, color='purple')
    plt.xlabel('Genres',color='white')
    plt.ylabel('Probabilities', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.gca().set_facecolor('none')
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    return render_template('analysis_results.html', plot_url=plot_url, file_id = prediction_list['file_id'], genres = genre_dict, features = rounded_feature_list)

def predict(file_path):
    sigmoid = torch.nn.Sigmoid()
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
    return prediction_data

def extract_audio(file):
    filename = file.filename
    audio = AudioSegment.from_file(file)
    start_time = (len(audio) / 2) - (30 * 1000) / 2 
    start_time = max(start_time, 0)
    refactored_audio = audio[start_time:start_time + (30 * 1000)]
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    refactored_audio.export(file_path, format="mp3")
    return(file_path)

def serialize_dict(data):
    if isinstance(data, dict):
        return {key: serialize_dict(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (int, float, str, bool)):
        return data
    else:
        return str(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.run(debug=True)
