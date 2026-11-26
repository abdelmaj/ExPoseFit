import io
from typing import List, Tuple

from constants import *
from datetime import datetime
from scipy.signal import savgol_filter
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import math
from scipy.ndimage import gaussian_filter



def trim_list_to_third_in_size(lst):
    if len(lst) < 3:
        return []
    step = len(lst) // 3 if len(lst) % 3 == 0 else len(lst) // 3 + 1
    return lst[step - 1::step]


def trim_list_uniformly(input_list, target_size=30):
    """
    Trims the given list to a specified size by selecting elements uniformly.

    :param input_list: List of elements to trim.
    :param target_size: Desired size of the output list.
    :return: A new list with `target_size` elements, uniformly selected.
    """
    if len(input_list) <= target_size:
        
        return input_list

    step = len(input_list) / target_size
    trimmed_list = [input_list[int(i * step)] for i in range(target_size)]
    return trimmed_list


def calculate_angle(a, b, c):
    """ Calculate the angle between three points: a, b, c
    Points should be in the form of (x, y) or (y, x) coordinates.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    
    ab = a - b
    cb = c - b

    
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def smooth_data(data, window_length=8):
    try:
        return savgol_filter(data, window_length=window_length, polyorder=2)
    except Exception as error:
        pass
        return data



















def read_video_frames(video_path: str) -> Tuple[List[cv2.typing.MatLike], int, float]:
    now = datetime.now()
    pass
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        raise EOFError(f"Error: Could not open video: {video_path}")

    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = vid.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        frames_remaining, frame = vid.read()
        if frames_remaining == False:
            break
        frames.append(frame)

    vid.release()
    pass
    return [frames, frame_count, frame_rate]


def resize_with_pad_cv2(image, target_height, target_width):
    image = np.array(image[0])
    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * scale), int(original_height * scale))
    resized_image = cv2.resize(
        image, new_size, interpolation=cv2.INTER_NEAREST)
    delta_w = target_width - new_size[0]
    delta_h = target_height - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return np.array([new_image])


def split_at_minima(lst):
    """
    Splits a list into two parts at the first occurrence of the minimum value.

    Args:
        lst (list): The list to split.

    Returns:
        tuple: A tuple containing two lists. The first includes elements
               up to and including the minima, the second starts after the minima.
    """
    
    min_index = lst.index(min(lst))

    
    part1 = lst[:min_index + 1]
    part2 = lst[min_index + 1:]

    return part1, part2


def count_line_intersections(values: list, line, filename: str = None) -> int:
    if filename is None:
        filename = str(math.floor(random.random() * 2**32)) + ".png"
    save_list_plot(values, line_y=line, filename="plots/" + filename)
    """
    Counts the number of intersections of a horizontal line with the plot of values.

    Args:
    values (list): A list of y-values.
    line (float): The y-value of the horizontal line.

    Returns:
    int: The number of intersections.
    """
    
    if len(values) == 0:
        return 0

    intersections = 0

    for i in range(len(values) - 1):
        
        if (values[i] < line < values[i + 1]) or (values[i] > line > values[i + 1]) \
                or line == values[i] or line == values[i + 1]:
            intersections += 1

    return intersections


def save_list_plot(values: list, line_y: float = None, filename: str = None):
    if filename is None:
        filename = str(math.floor(random.random() * 2**32)) + ".png"
    """
    Saves the plot of a list of values to a file.

    Args:
    values (list): List of values to plot.
    filename (str): The filename to save the plot (e.g., "plot.png").
    line_y (float, optional): If provided, a horizontal line at this y-value will be added to the plot.
    """
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(values, label="Values")

    
    if line_y is not None:
        plt.axhline(y=line_y, color='red', linestyle='--',
                    label=f'Line at y={line_y}')

    plt.title("Plot of Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    
    plt.savefig(filename)
    plt.close()  


def remove_anomalies_and_interpolate(data, threshold=2.0, window_size=3):
    """
    Detects anomalies in a list using a rolling window, removes them, and interpolates missing values.

    Args:
        data (list): The input list of numeric data.
        window_size (int): The size of the rolling window to calculate local mean and std.
        threshold (float): The number of standard deviations from the mean to classify anomalies.

    Returns:
        tuple: (list of cleaned and interpolated data, list of anomaly indices)
    """
    data = np.array(data, dtype=float)  
    anomalies = []

    
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        
        window = np.concatenate((data[start_idx:i], data[i + 1:end_idx]))

        if len(window) < 2:  
            continue

        mean = np.mean(window)
        std_dev = np.std(window)

        if abs(data[i] - mean) > threshold * std_dev:
            anomalies.append(i)

    
    for idx in anomalies:
        data[idx] = np.nan

    
    nan_indices = np.isnan(data)
    valid_indices = ~nan_indices
    data[nan_indices] = np.interp(np.flatnonzero(
        nan_indices), np.flatnonzero(valid_indices), data[valid_indices])

    return data.tolist()


def gaussian_smoothing(data, sigma=3):
    return gaussian_filter(data, sigma=sigma)

def determine_highest_confidence_for_lower(keypoints):
    """
    Determines which leg (left or right) provides more accurate keypoint confidence data 
    for the HIP,KNEE and ANKLE joints from MovNet tensor data across frames.

    Args:
        keypoints (list): A list of tensors, where each tensor represents keypoint data for a frame.
                          Each tensor is expected to contain confidence scores for various body joints.

    Returns:
        str: "right" if the right leg's combined confidence scores are higher, 
             "left" otherwise.
    """
    right_confidence, left_confidence = 0, 0
    
    for keypoint in keypoints:
        
        right_confidence += (
        keypoint[Keypoints.right_hip] +
        keypoint[Keypoints.right_knee] +
        keypoint[Keypoints.right_ankle])[2]

        
        left_confidence += (
        keypoint[Keypoints.left_hip] +
        keypoint[Keypoints.left_knee] +
        keypoint[Keypoints.left_ankle])[2]

    pass
    pass
    
    return "right" if right_confidence > left_confidence else "left"


def determine_highest_confidence_for_upper(keypoints):
    """
    Determines which leg (left or right) provides more accurate keypoint confidence data 
    for the SHOULDER,ELBOW and WRIST joints from MovNet tensor data across frames.

    Args:
        keypoints (list): A list of tensors, where each tensor represents keypoint data for a frame.
                          Each tensor is expected to contain confidence scores for various body joints.

    Returns:
        str: "right" if the right leg's combined confidence scores are higher, 
             "left" otherwise.
    """
    right_confidence, left_confidence = 0, 0
    
    for keypoint in keypoints:
        
        right_confidence += (
        keypoint[Keypoints.right_elbow] +
        keypoint[Keypoints.right_shoulder] +
        keypoint[Keypoints.right_wrist])[2]

        
        left_confidence += (
        keypoint[Keypoints.left_elbow] +
        keypoint[Keypoints.left_shoulder] +
        keypoint[Keypoints.left_wrist])[2]


    pass
    pass
    
    return "right" if right_confidence > left_confidence else "left"