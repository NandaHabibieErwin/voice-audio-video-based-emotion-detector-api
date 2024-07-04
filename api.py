from flask import Flask, request, jsonify
import torch
import os
import numpy as np
import source.audio_analysis_utils.model as audio_model
import source.audio_analysis_utils.predict as audio_predict

import source.face_emotion_utils.model as face_model
import source.face_emotion_utils.predict as face_predict
import source.face_emotion_utils.utils as face_utils
from source import config
import source.face_emotion_utils.preprocess_main as face_preprocess_main

import source.audio_face_combined.model as combined_model
import source.audio_face_combined.preprocess_main as combined_data
import source.audio_face_combined.combined_config as combined_config
import source.audio_face_combined.predict as combined_predict
import source.audio_face_combined.download_video as download_youtube
import source.audio_face_combined.utils as combined_utils

from flask_cors import CORS

import cv2
import sys
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)


def make_serializable(result):
    if isinstance(result, (np.floating, float)):
        return float(result)
    elif isinstance(result, (np.integer, int)):
        return int(result)
    elif isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, torch.Tensor):
        return result.cpu().numpy().tolist()
    else:
        return result

@app.route('/upload-image', methods=['POST'])
def upload_image():
    logging.debug('upload-image endpoint called')
    image_file = request.files['file']
    file_name = image_file.filename
    file_path = os.path.join('input_files', file_name)
    image_file.save(file_path)
    logging.debug(f'File saved to {file_path}')
    file_name = face_utils.find_filename_match(known_filename=file_name, directory=config.INPUT_FOLDER_PATH)
    print("file_name", file_name)
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # emotion, confidence, softmax_probs_string = face_predict.predict(image)
    prediction, confidence, softmax_probs, _ = face_predict.predict(image)
    
    # Convert softmax probabilities to string if needed
    softmax_probs_string = ','.join(map(str, softmax_probs))
    confidence = float(confidence)

    
    # Return prediction result
    return jsonify({
        'emotion': prediction,
        'confidence': confidence,
        'softmax_probs_string': softmax_probs_string
    })

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    # Handle file upload and prediction
    logging.debug('upload-audio endpoint called')
    audio_file = request.files['file']
    logging.debug('File received')
    # Save file temporarily
    file_path = os.path.join('input_files', audio_file.filename)
    audio_file.save(file_path)
    logging.debug(f'File saved to {file_path}')
    audio_name = audio_file.filename
    # Perform prediction
    emotion, confidence, softmax_probs_string = audio_predict.predict(audio_name)
    logging.debug('Prediction completed')
    
    # Ensure all parts of the result are JSON serializable
    emotion = make_serializable(emotion)
    confidence = make_serializable(confidence)
    softmax_probs_string = make_serializable(softmax_probs_string)
    
    # Return prediction result
    return jsonify({'emotion': emotion, 'confidence': confidence, 'softmax_probs_string': softmax_probs_string})

@app.route('/upload-video', methods=['POST'])
def upload_video():
    logging.debug('upload-video endpoint called')
    video_file = request.files['file']
    file_path = os.path.join('input_files', video_file.filename)
    video_file.save(file_path)
    logging.debug(f'File saved to {file_path}')
    
    # Perform prediction
    video_name = face_utils.find_filename_match(video_file.filename, config.INPUT_FOLDER_PATH)
    prediction_result = combined_predict.predict_video(file_path)
    
    # Extract necessary details
    transcript = prediction_result['transcript']
    confidence = prediction_result['confidence']
    softmax_probs_string = prediction_result['softmax_probs_string']
    
    # Find the most predicted emotion and its probability
    predictions = prediction_result['predictions']
    max_prob_index = np.argmax(prediction_result['sum_probs'])
    emotion = config.EMOTION_INDEX[max_prob_index]
    confidence = round(prediction_result['sum_probs'][max_prob_index] / len(predictions) * 100, 2)
    
    # Return prediction result
    return jsonify({
        'emotion': emotion,
        'confidence': confidence,
        'softmax_probs_string': softmax_probs_string,
        'transcript': transcript
    })



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
