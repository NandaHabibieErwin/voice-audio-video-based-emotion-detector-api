import source.face_emotion_utils.utils as face_utils
import source.face_emotion_utils.face_mesh as face_mesh
import source.face_emotion_utils.face_config as face_config

import source.audio_analysis_utils.utils as audio_utils

import source.pytorch_utils.visualize as pt_vis

import source.config as config

import cv2
import numpy as np
from PIL import Image as ImagePIL
import time
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

FACE_SQUARE_SIZE = 64


def _create_gradcam(model, model_input, target_layer, device, verbose=False):
    return pt_vis.create_gradcam(model, model_input, target_layer, device, FACE_SQUARE_SIZE, verbose=verbose)


def _overlay_gradcam_on_image(img, grad_cam_pil, alpha=0.5, square_size=FACE_SQUARE_SIZE):
    return pt_vis.overlay_gradcam_on_image(img, grad_cam_pil, alpha=alpha, square_size=square_size)


def _visualise_feature_maps(feature_map, feature_map_name):
    pt_vis.visualise_feature_maps(feature_map, feature_map_name)


def _get_prediction(
        best_hp,
        img,
        model,
        imshow=False,
        video_mode=False,
        grad_cam=True,
        grad_cam_on_video=False,
        feature_maps_flag=True,
        device=config.device,
        verbose=True,
        emotion_index_dict=config.EMOTION_INDEX,
):
    try:
        # We detect the face and get the landmarks, regardless of if landmarks are used or not. This is because we need the face image for the model input
        result = face_mesh.get_mesh(image=cv2.cvtColor(img, cv2.COLOR_RGB2BGR), upscale_landmarks=True, showImg=False, print_flag=True, return_mesh=True)
    except:
        raise Exception("Face mesh failed")
    if result is None:
        if verbose:
            print("No face detected")
        return None

    landmarks_depths, face_input_org, annotated_image, (tl_xy, br_xy) = result

    # Normalise if needed
    normalise = best_hp['normalise']
    if normalise:
        landmarks_depths = face_utils.normalise_lists([landmarks_depths], save_min_max=True, print_flag=verbose)[0]

    landmarks_depths = np.array(landmarks_depths)

    # Get the full image
    face_input = cv2.cvtColor(face_input_org, cv2.COLOR_BGR2GRAY)
    face_input = cv2.resize(face_input, (face_config.FACE_SIZE, face_config.FACE_SIZE))

    # Prep it for pytorch
    face_input = np.repeat(face_input[np.newaxis, :, :], 3, axis=0)
    if verbose:
        print("face_input.shape", face_input.shape)
    x = np.array(face_input)
    x = x / 255.
    x = x.reshape(face_utils.get_input_shape("image"))
    x = np.array(x[np.newaxis, :])
    if verbose:
        print(x.shape)

    landmarks_depths = np.array(landmarks_depths[np.newaxis, :])
    if verbose:
        print(landmarks_depths.shape)

    model_input = (x, landmarks_depths)

    # Get the prediction from the model
    pred = model(torch.from_numpy(np.array(model_input[0])).float().to(device),
                 torch.from_numpy(np.array(model_input[1])).float().to(device))
    pred = torch.nn.functional.softmax(pred, dim=1)
    if verbose:
        print("NN output:\n", pred)

    # Organise the prediction
    prediction_index = int(list(pred[0]).index(max(pred[0])))
    pred_numpy = pred[0].detach().cpu().numpy()

    if verbose:
        print("\nPrediction index: ", prediction_index)
        print("Prediction label: ", emotion_index_dict[prediction_index])
        print("Prediction probability: ", max(pred_numpy))
        print("\n\nPrediction probabilities:\n", audio_utils.get_softmax_probs_string(pred_numpy, list(emotion_index_dict.values())))

    string = audio_utils.get_softmax_probs_string(pred_numpy, list(emotion_index_dict.values()))
    string_img = emotion_index_dict[prediction_index] + ": " + str(round(max(pred_numpy) * 100)) + "%"

    return_objs = (emotion_index_dict[prediction_index], prediction_index, list(pred_numpy), img)

    return return_objs


def predict(
        image=None,
        video_mode=False,
        webcam_mode=False,
        model_save_path=config.FACE_MODEL_SAVE_PATH,
        best_hp_json_path=config.FACE_BEST_HP_JSON_SAVE_PATH,
        verbose=face_config.PREDICT_VERBOSE,
        imshow=face_config.SHOW_PRED_IMAGE,
        grad_cam=face_config.GRAD_CAM,
        grad_cam_on_video=face_config.GRAD_CAM_ON_VIDEO,
):

    best_hyperparameters = face_utils.load_dict_from_json(best_hp_json_path)
    if verbose:
        print(f"Best hyperparameters, {best_hyperparameters}")

    model = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.to(config.device).eval()

    if type(image) == str:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = _get_prediction(best_hp=best_hyperparameters, img=image, model=model, imshow=imshow, video_mode=video_mode, verbose=verbose, grad_cam=grad_cam)

    if verbose:
        print("\n\n\nResults:")
        for res in result:
            # check if numpy
            if type(res) == np.ndarray:
                print(res.shape)
            else:
                print(res)

    emotion_name, emotion_index, prediction_probabilities = result[0], result[1], result[2]
    prediction = emotion_name
    confidence = max(prediction_probabilities)
    softmax = np.exp(prediction_probabilities) / np.sum(np.exp(prediction_probabilities))
    softmax = [float(s) for s in softmax]

    # Return only the required values along with the original result tuple
    return prediction, confidence, softmax, result


