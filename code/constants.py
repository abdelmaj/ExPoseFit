THUNDER_MOVENET_IMAGE_INPUT_SIZE = 256
KEYPOINT_CONFIDENCE_THRESHOLD = 0.11
PERSON_PRESENT_THRESHOLD = 0.2
KINETICS_400_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
KINETICS_600_URL = "https://raw.githubusercontent.com/google-deepmind/kinetics-i3d/refs/heads/master/data/label_map_600.txt"


class Keypoints:
    nose = 0, 0, 0,
    left_eye = 0, 0, 1,
    right_eye = 0, 0, 2,
    left_ear = 0, 0, 3,
    right_ear = 0, 0, 4,
    left_shoulder = 0, 0, 5,
    right_shoulder = 0, 0, 6,
    left_elbow = 0, 0, 7,
    right_elbow = 0, 0, 8,
    left_wrist = 0, 0, 9,
    right_wrist = 0, 0, 10,
    left_hip = 0, 0, 11,
    right_hip = 0, 0, 12,
    left_knee = 0, 0, 13,
    right_knee = 0, 0, 14,
    left_ankle = 0, 0, 15,
    right_ankle = 0, 0, 16,
