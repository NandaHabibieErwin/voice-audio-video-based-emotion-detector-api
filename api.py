from flask import Flask, request, jsonify
import source.audio_analysis_utils.predict as audio_predict

import torch
import os
import numpy as np
import source.audio_analysis_utils.model as audio_model
import source.audio_analysis_utils.predict as audio_predict

from flask_cors import CORS

import cv2
import sys
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)


def make_serializable(result):
    if isinstance(result, (np.float32, np.float64)):
        return float(result)
    elif isinstance(result, (np.int32, np.int64)):
        return int(result)
    elif isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, torch.Tensor):
        return result.cpu().numpy().tolist()
    else:
        return result

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
    video_file = request.files['file']
    file_path = os.path.join('input_files', video_file.filename)
    logging.debug(f'File saved to {file_path}')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
