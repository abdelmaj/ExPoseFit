from typing import Tuple
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import os
from datetime import datetime
import matplotlib.patches as patches
from util import resize_with_pad_cv2, read_video_frames, trim_list_uniformly
from constants import THUNDER_MOVENET_IMAGE_INPUT_SIZE, PERSON_PRESENT_THRESHOLD, KEYPOINT_CONFIDENCE_THRESHOLD

print("MoveNet: Downloading Thunder Model...")
thunder_model = hub.load(
    "https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4"
)
print("MoveNet: Downloaded Thunder Model which took")

movenet = thunder_model.signatures['serving_default']


def extract_keypoints_from_frame(frame: cv2.typing.MatLike):
    
    prepare_img = datetime.now()

    now = datetime.now()
    image = tf.expand_dims(frame, axis=0)
    
    
    now = datetime.now()

    image = tf.cast(resize_with_pad_cv2(
        image, THUNDER_MOVENET_IMAGE_INPUT_SIZE, THUNDER_MOVENET_IMAGE_INPUT_SIZE,
    ), dtype=tf.int32)
    
    

    
    

    now = datetime.now()
    movenet_output = movenet(image)
    
    

    return [movenet_output['output_0'], image]


def extract_keypoints_from_video(video_path: str):
    print("extract_keypoints_from_video called")
    now = datetime.now()
    [video_frames, _, _] = read_video_frames(video_path)
    video_frames = video_frames
    
    video_keypoints = []
    for frame in video_frames:
        [keyframes, _] = extract_keypoints_from_frame(frame)
        video_keypoints.append(keyframes)
    print(
        f"extract_keypoints_from_video took {datetime.now() - now} to extract {len(video_frames)} frames")
    return video_keypoints


def is_person_present(keypoints) -> bool:
    """
    Should only take about 50ms
    """
    confidences = np.array(keypoints)[..., 2].flatten()
    sum_of_all_confidences = np.sum(confidences)
    score = float(sum_of_all_confidences /
                  confidences.shape[0])
    print("is_person_present score " + str(score))
    return score > PERSON_PRESENT_THRESHOLD


def is_full_body_visible(keypoints) -> bool:
    """
    Determines if a person is completely present by counting the number of keypoints which
    exceed the agreed upon threshold then checks if percentage of these points is larger than 90%
    Should only take about 50ms.
    """
    confidences = np.array(keypoints)[..., 2].flatten()
    num_of_keypoints_with_needed_confidence = np.sum(
        confidences > KEYPOINT_CONFIDENCE_THRESHOLD)
    return float(num_of_keypoints_with_needed_confidence / confidences.shape[0]) > 0.7
