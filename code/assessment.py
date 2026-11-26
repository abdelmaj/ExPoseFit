import numpy as np
from constants import Keypoints
from util import calculate_angle, split_at_minima, count_line_intersections,smooth_data, remove_anomalies_and_interpolate ,gaussian_smoothing, determine_highest_confidence_for_lower,determine_highest_confidence_for_upper
from datetime import datetime
import random
import math
import tensorflow as tf


def extract_base_ankles_angles(keypoints):
    """
    Extract the angles for BASE and the two ANKLEs from MovNet tensor data.
    keypoints: List of tensors where each tensor contains keypoint data for a frame.
    Returns a list of angles for each frame.
    """
    angles = []

    k1 = Keypoints.left_hip 
    k3 = Keypoints.left_ankle 
    k4 = Keypoints.right_hip   
    k6 = Keypoints.right_ankle

    for keypoint in keypoints:
        
        baseX = (keypoint[k1[0],k1[1],k1[2], 0] + keypoint[k4[0],k4[1],k4[2], 0]) / 2
        baseY = (keypoint[k1[0],k1[1],k1[2], 1] + keypoint[k4[0],k4[1],k4[2], 1]) / 2

        base = tf.stack([baseX, baseY])

        right_ankle = keypoint[k6[0],k6[1],k6[2], :2]
        left_ankle = keypoint[k3[0],k3[1],k3[2], :2]

        
        angle = calculate_angle(base, left_ankle, right_ankle)   
        angles.append(angle)     

    return angles


def extract_pullup_angles(frames):
    """
    Extract the angles for SHOULDER,ELBOW and WRIST from MovNet tensor data.
    frames: List of tensors where each tensor contains keypoint data for a frame.
    Returns a list of angles for each arm.
    """
    right_angles = []
    left_angles = []

    for frame in frames:
        
        left_shoulder = frame[Keypoints.left_shoulder[0], Keypoints.left_shoulder[1], Keypoints.left_shoulder[2], :2] 
        left_elbow = frame[Keypoints.left_elbow[0], Keypoints.left_elbow[1], Keypoints.left_elbow[2], :2]
        left_wrist = frame[Keypoints.left_wrist[0],  Keypoints.left_wrist[1], Keypoints.left_wrist[2], :2]
        
        right_shoulder = frame[Keypoints.right_shoulder[0], Keypoints.right_shoulder[1], Keypoints.right_shoulder[2], :2] 
        right_elbow = frame[Keypoints.right_elbow[0], Keypoints.right_elbow[1], Keypoints.right_elbow[2], :2]
        right_wrist = frame[Keypoints.right_wrist[0],  Keypoints.right_wrist[1], Keypoints.right_wrist[2], :2]

        
        left_angle = calculate_angle(left_elbow, left_shoulder, left_wrist)
        right_angle = calculate_angle(right_elbow, right_shoulder, right_wrist)
        
        right_angles.append(right_angle)
        left_angles.append(left_angle)

    return right_angles , left_angles


def extract_lunges_angles(keypoints):
    """
    Extract the angles for HIP,KNEE and ANKLE from MovNet tensor data.
    keypoints: List of tensors where each tensor contains keypoint data for a frame.
    Returns a list of angles for each leg. 
    """
    right_leg_angles = []
    left_leg_angles = []

    for keypoint in keypoints:
        
        right_hip = keypoint[Keypoints.right_hip[0], Keypoints.right_hip[1], Keypoints.right_hip[2], :2] 
        right_knee = keypoint[Keypoints.right_knee[0], Keypoints.right_knee[1], Keypoints.right_knee[2], :2]
        right_ankle = keypoint[Keypoints.right_ankle[0], Keypoints.right_ankle[1], Keypoints.right_ankle[2], :2]

        left_hip = keypoint[Keypoints.left_hip[0], Keypoints.left_hip[1], Keypoints.left_hip[2], :2] 
        left_knee = keypoint[Keypoints.left_knee[0], Keypoints.left_knee[1], Keypoints.left_knee[2], :2]
        left_ankle = keypoint[Keypoints.left_ankle[0], Keypoints.left_ankle[1], Keypoints.left_ankle[2], :2]

        
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        right_leg_angles.append(right_angle)
        left_leg_angles.append(left_angle)

    return right_leg_angles , left_leg_angles

def extract_squat_angles_confidence_aware(keypoints, conf_thresh=0.3):
    left_angles = []
    right_angles = []

    for kp in keypoints:
        
        lh = kp[Keypoints.left_hip[0], Keypoints.left_hip[1], Keypoints.left_hip[2], :]
        lk = kp[Keypoints.left_knee[0], Keypoints.left_knee[1], Keypoints.left_knee[2], :]
        la = kp[Keypoints.left_ankle[0], Keypoints.left_ankle[1], Keypoints.left_ankle[2], :]

        
        rh = kp[Keypoints.right_hip[0], Keypoints.right_hip[1], Keypoints.right_hip[2], :]
        rk = kp[Keypoints.right_knee[0], Keypoints.right_knee[1], Keypoints.right_knee[2], :]
        ra = kp[Keypoints.right_ankle[0], Keypoints.right_ankle[1], Keypoints.right_ankle[2], :]

        left_conf = float(lh[2] + lk[2] + la[2]) / 3.0
        right_conf = float(rh[2] + rk[2] + ra[2]) / 3.0

        left_angle = None
        right_angle = None

        if left_conf >= conf_thresh:
            left_angle = calculate_angle([lh[1], lh[0]], [lk[1], lk[0]], [la[1], la[0]])

        if right_conf >= conf_thresh:
            right_angle = calculate_angle([rh[1], rh[0]], [rk[1], rk[0]], [ra[1], ra[0]])

        if left_angle is not None and right_angle is not None:
            if left_conf >= right_conf:
                left_angles.append(left_angle)
            else:
                left_angles.append(right_angle)
        elif left_angle is not None:
            left_angles.append(left_angle)
        elif right_angle is not None:
            left_angles.append(right_angle)

    return left_angles

def extract_pushup_angles_confidence_aware(keypoints, conf_thresh=0.3):
    selected_angles = []

    for kp in keypoints:
        ls = kp[Keypoints.left_shoulder[0], Keypoints.left_shoulder[1], Keypoints.left_shoulder[2], :]
        le = kp[Keypoints.left_elbow[0], Keypoints.left_elbow[1], Keypoints.left_elbow[2], :]
        lw = kp[Keypoints.left_wrist[0], Keypoints.left_wrist[1], Keypoints.left_wrist[2], :]

        rs = kp[Keypoints.right_shoulder[0], Keypoints.right_shoulder[1], Keypoints.right_shoulder[2], :]
        re = kp[Keypoints.right_elbow[0], Keypoints.right_elbow[1], Keypoints.right_elbow[2], :]
        rw = kp[Keypoints.right_wrist[0], Keypoints.right_wrist[1], Keypoints.right_wrist[2], :]

        left_conf = float(ls[2] + le[2] + lw[2]) / 3.0
        right_conf = float(rs[2] + re[2] + rw[2]) / 3.0

        left_angle = None
        right_angle = None

        if left_conf >= conf_thresh:
            left_angle = calculate_angle([ls[1], ls[0]], [le[1], le[0]], [lw[1], lw[0]])

        if right_conf >= conf_thresh:
            right_angle = calculate_angle([rs[1], rs[0]], [re[1], re[0]], [rw[1], rw[0]])

        if left_angle is not None and right_angle is not None:
            selected_angles.append(left_angle if left_conf >= right_conf else right_angle)
        elif left_angle is not None:
            selected_angles.append(left_angle)
        elif right_angle is not None:
            selected_angles.append(right_angle)

    return selected_angles


'''
def extract_pushup_angles(keypoints):
    """
    Extract the angles for SHOULDER,ELBOW and WRIST from MovNet tensor data.
    keypoints: List of tensors where each tensor contains keypoint data for a frame.
    Returns the list of angles across all frames. 
    """
    a = Keypoints.left_shoulder
    b = Keypoints.left_elbow
    c = Keypoints.left_wrist

    shoulder_keypoints = np.array(keypoints)[:, a[0], a[1], a[2], :2]
    elbow_keypoints = np.array(keypoints)[:, b[0], b[1], b[2], :2]
    wrist_keypoints = np.array(keypoints)[:, c[0], c[1], c[2], :2]

    shoulder_elbow_wrist_angles = []

    for i, cord in enumerate(shoulder_keypoints):
        p1 = [shoulder_keypoints[i][1], shoulder_keypoints[i][0]]
        p2 = [elbow_keypoints[i][1], elbow_keypoints[i][0]]
        p3 = [wrist_keypoints[i][1], wrist_keypoints[i][0]]

        shoulder_elbow_wrist_angles.append(calculate_angle(p1, p2, p3))

    return shoulder_elbow_wrist_angles
'''

def extract_situp_angles(keypoints):
    """
    Extract sit-up joint-angle trajectories.

    Returns:
        1. shoulder-hip-knee angle (vertex at hip)
        2. hip-knee-ankle angle (vertex at knee)
        3. hip coordinates
    """

    
    keypoints = [
        kp.numpy() if hasattr(kp, "numpy") else np.asarray(kp)
        for kp in keypoints
    ]

    keypoints = np.asarray(keypoints)

    
    
    
    

    hip = Keypoints.left_hip
    knee = Keypoints.left_knee
    shoulder = Keypoints.left_shoulder

    hip_pts = keypoints[
        :, hip[0], hip[1], hip[2], :2
    ]

    knee_pts = keypoints[
        :, knee[0], knee[1], knee[2], :2
    ]

    shoulder_pts = keypoints[
        :, shoulder[0], shoulder[1], shoulder[2], :2
    ]

    shoulder_hip_knee_angles = []
    hip_cords = []

    for i in range(len(keypoints)):

        p_shoulder = [
            shoulder_pts[i][1],
            shoulder_pts[i][0]
        ]

        p_hip = [
            hip_pts[i][1],
            hip_pts[i][0]
        ]

        p_knee = [
            knee_pts[i][1],
            knee_pts[i][0]
        ]

        hip_cords.append(p_hip)

        
        angle = calculate_angle(
            p_shoulder,
            p_hip,
            p_knee
        )

        shoulder_hip_knee_angles.append(
            angle
        )

    
    
    
    

    ankle = Keypoints.left_ankle

    ankle_pts = keypoints[
        :, ankle[0], ankle[1], ankle[2], :2
    ]

    hip_knee_ankle_angles = []

    for i in range(len(keypoints)):

        p_hip = [
            hip_pts[i][1],
            hip_pts[i][0]
        ]

        p_knee = [
            knee_pts[i][1],
            knee_pts[i][0]
        ]

        p_ankle = [
            ankle_pts[i][1],
            ankle_pts[i][0]
        ]

        
        angle = calculate_angle(
            p_hip,
            p_knee,
            p_ankle
        )

        hip_knee_ankle_angles.append(
            angle
        )

    return [
        remove_anomalies_and_interpolate(
            shoulder_hip_knee_angles,
            0.8
        ),
        hip_knee_ankle_angles,
        hip_cords
    ]

def extract_situp_hip_abduction_from_keypoints(keypoints):
    k3 = Keypoints.left_knee
    k4 = Keypoints.right_knee

    c = np.array(keypoints)[:, k3[0], k3[1], k3[2], :2]
    d = np.array(keypoints)[:, k4[0], k4[1], k4[2], :2]

    angles = []
    xcords = []
    for i, cord in enumerate(c):
        angles.append(abs(d[i][0] - c[i][0]))
        xcords.append(d[i][0])

    return [angles, xcords]

'''
def count_squat_reps_hysteresis(angles, down_th=100, up_th=140):
    state = "UP"
    reps = 0

    for a in angles:
        if state == "UP" and a < down_th:
            state = "DOWN"
        elif state == "DOWN" and a > up_th:
            state = "UP"
            reps += 1

    return reps
  '''

def asses_form_of_squat(keypoints):
    keypoints = [
        kp.numpy() if hasattr(kp, "numpy") else np.asarray(kp)
        for kp in keypoints
    ]
    print("asses_form_of_squat called")

    angles = extract_squat_angles_confidence_aware(keypoints)

    if len(angles) == 0:
        return [0, ["Pose detection is unreliable for this video"]]

    angles = gaussian_smoothing(angles)

    min_a = min(angles)
    max_a = max(angles)
    motion_range = max_a - min_a

    if motion_range < 20:
        return [0, ["Not enough squat motion detected"]]

    down_th = min_a + 0.35 * motion_range
    up_th = min_a + 0.75 * motion_range

    feedback = []
    valid_reps = 0
    invalid_reps = 0

    state = "UP"
    rep_bottom = float("inf")
    min_frames_between_reps = 8
    frames_since_last_rep = 999

    for i, a in enumerate(angles):
        frames_since_last_rep += 1

        if state == "UP":
            if a < down_th:
                state = "DOWN"
                rep_bottom = a
                print(f"Frame {i}: started squat, angle={a:.2f}")

        elif state == "DOWN":
            rep_bottom = min(rep_bottom, a)

            if a > up_th and frames_since_last_rep >= min_frames_between_reps:
                if rep_bottom < down_th:
                    valid_reps += 1
                    print(f"Frame {i}: valid rep #{valid_reps}, bottom={rep_bottom:.2f}, top={a:.2f}")
                else:
                    invalid_reps += 1
                    print(f"Frame {i}: invalid rep, bottom={rep_bottom:.2f}, top={a:.2f}")

                state = "UP"
                rep_bottom = float("inf")
                frames_since_last_rep = 0

    if valid_reps > 0:
        feedback.append("You're doing a squat")
    if invalid_reps > 0:
        feedback.append("Some repetitions were incomplete")

    return [valid_reps, feedback]

def asses_form_of_pushup(keypoints):
    keypoints = [
        kp.numpy() if hasattr(kp, "numpy") else np.asarray(kp)
        for kp in keypoints
    ]
    print("asses_form_of_pushup called")

    angles = extract_pushup_angles_confidence_aware(keypoints)

    if len(angles) == 0:
        return [0, ["Pose detection is unreliable for this video"]]

    angles = gaussian_smoothing(angles)

    print("\nSmoothed pushup angles")
    print(angles)
    print("\n")

    min_a = min(angles)
    max_a = max(angles)
    motion_range = max_a - min_a

    if motion_range < 15:
        return [0, ["Not enough pushup motion detected"]]

    
    
    
    down_th = min_a + 0.35 * motion_range
    up_th = min_a + 0.75 * motion_range

    print(f"min_a={min_a:.2f}, max_a={max_a:.2f}, down_th={down_th:.2f}, up_th={up_th:.2f}")

    feedback = []
    valid_reps = 0
    shallow_reps = 0
    incomplete_up_reps = 0

    state = "UP"
    rep_bottom = float("inf")
    current_top = angles[0]

    min_frames_between_reps = 8
    frames_since_last_rep = 999

    for a in angles:
        frames_since_last_rep += 1

        if state == "UP":
            current_top = max(current_top, a)

            
            if a < down_th:
                state = "DOWN"
                rep_bottom = a

        elif state == "DOWN":
            rep_bottom = min(rep_bottom, a)

            
            if a > up_th and frames_since_last_rep >= min_frames_between_reps:
                went_low_enough = rep_bottom < down_th
                came_up_enough = a > up_th

                if went_low_enough and came_up_enough:
                    valid_reps += 1

                state = "UP"
                current_top = a
                rep_bottom = float("inf")
                frames_since_last_rep = 0

    
    for i in range(1, len(angles) - 1):
        if angles[i] < angles[i - 1] and angles[i] < angles[i + 1]:
            local_min = angles[i]

            left_peak = max(angles[max(0, i - 12):i]) if i > 0 else angles[i]
            right_peak = max(angles[i + 1:min(len(angles), i + 13)]) if i + 1 < len(angles) else angles[i]

            
            if (left_peak - local_min > 10) and (right_peak - local_min > 10) and local_min >= down_th:
                shallow_reps += 1

            
            elif local_min < down_th and right_peak <= up_th:
                incomplete_up_reps += 1

    if valid_reps > 0:
        feedback.append("You're doing a pushup")
    if shallow_reps > 0:
        feedback.append("Go lower on some repetitions")
    if incomplete_up_reps > 0:
        feedback.append("Extend your arms more at the top on some repetitions")
    if valid_reps == 0 and shallow_reps == 0 and incomplete_up_reps == 0:
        feedback.append("No valid pushup repetitions detected")

    print(f"valid_reps={valid_reps}, shallow_reps={shallow_reps}, incomplete_up_reps={incomplete_up_reps}")

    return [valid_reps, feedback]

def asses_form_of_situps(keypoints):
    [shoulder_hip_knee_angles, hip_knee_ankle_angles, _] = extract_situp_angles(keypoints)
    print("asses_form_of_situps called")

    if len(shoulder_hip_knee_angles) == 0:
        return [0, ["Pose detection is unreliable for this video"]]

    angles = gaussian_smoothing(shoulder_hip_knee_angles)

    print("\nSmoothed situp angles")
    print(angles)
    print("\n")

    min_a = min(angles)
    max_a = max(angles)
    motion_range = max_a - min_a

    if motion_range < 12:
        return [0, ["Not enough situp motion detected"]]

    
    
    
    down_th = min_a + 0.3 * motion_range
    up_th = min_a + 0.7 * motion_range

    print(f"min_a={min_a:.2f}, max_a={max_a:.2f}, down_th={down_th:.2f}, up_th={up_th:.2f}")

    HIP_KNEE_ANKLE_MIN = 20
    HIP_KNEE_ANKLE_MAX = 80

    feedback = []
    valid_reps = 0
    shallow_reps = 0
    incomplete_up_reps = 0

    
    if any(x > HIP_KNEE_ANKLE_MAX for x in hip_knee_ankle_angles):
        feedback.append("Close your knees more")
    if any(x < HIP_KNEE_ANKLE_MIN for x in hip_knee_ankle_angles):
        feedback.append("Open your knees more")

    state = "UP"
    rep_bottom = float("inf")
    current_top = angles[0]

    min_frames_between_reps = 8
    frames_since_last_rep = 999

    for a in angles:
        frames_since_last_rep += 1

        if state == "UP":
            current_top = max(current_top, a)

            
            if a < down_th:
                state = "DOWN"
                rep_bottom = a

        elif state == "DOWN":
            rep_bottom = min(rep_bottom, a)

            
            if a > up_th and frames_since_last_rep >= min_frames_between_reps:
                went_low_enough = rep_bottom < down_th
                came_up_enough = a > up_th

                if went_low_enough and came_up_enough:
                    valid_reps += 1

                state = "UP"
                current_top = a
                rep_bottom = float("inf")
                frames_since_last_rep = 0

    
    for i in range(1, len(angles) - 1):
        if angles[i] < angles[i - 1] and angles[i] < angles[i + 1]:
            local_min = angles[i]

            left_peak = max(angles[max(0, i - 12):i]) if i > 0 else angles[i]
            right_peak = max(angles[i + 1:min(len(angles), i + 13)]) if i + 1 < len(angles) else angles[i]

            
            if (left_peak - local_min > 8) and (right_peak - local_min > 8) and local_min >= down_th:
                shallow_reps += 1

            
            elif local_min < down_th and right_peak <= up_th:
                incomplete_up_reps += 1

    if valid_reps > 0:
        feedback.append("You're doing a situp")
    if shallow_reps > 0:
        feedback.append("Get closer to your knees on some repetitions")
    if incomplete_up_reps > 0:
        feedback.append("Return more at the top on some repetitions")
    if valid_reps == 0 and shallow_reps == 0 and incomplete_up_reps == 0:
        feedback.append("No valid situp repetitions detected")

    print(f"valid_reps={valid_reps}, shallow_reps={shallow_reps}, incomplete_up_reps={incomplete_up_reps}")

    return [valid_reps, feedback]

'''
def asses_form_of_pushup(keypoints):
    print("asses_form_of_pushup called")
    filename = str(math.floor(random.random() * 2**32))

    shoulder_elbow_wrist_angles = extract_pushup_angles(keypoints)
    MIN_THRESHOLD = 90
    RETURN_THRESHOLD = 130
    feedback = []
    reps = 0
    # MAX_ANGLE_ALONG_HIPS = 25
    min_angle = min(shoulder_elbow_wrist_angles)

    if min_angle > MIN_THRESHOLD:
        feedback.append("Make sure to go down more")
        return [0, feedback]

    feedback.append("You're doing a pushup")
    shoulder_elbow_wrist_angles = smooth_data(shoulder_elbow_wrist_angles, 4)
    reps = min(count_line_intersections(shoulder_elbow_wrist_angles, MIN_THRESHOLD, filename=filename + "-pushup-min.png"),
               count_line_intersections(shoulder_elbow_wrist_angles, RETURN_THRESHOLD, filename=filename + "-pushup-return.png")) / 2

    return [reps, feedback]



def asses_form_of_situps(keypoints):
    [shoulder_hip_knee_angles,
        hip_knee_ankle_angles, _] = extract_situp_angles(keypoints)
    print("asses_form_of_situps called")
    MIN_THRESHOLD = 20
    RETURN_THRESHOLD = 50
    HIP_KNEE_ANKLE_MIN = 20
    HIP_KNEE_ANKLE_MAX = 80
    feedback = []
    reps = 0
    min_angle = min(shoulder_hip_knee_angles)

    if any(x > HIP_KNEE_ANKLE_MAX for x in hip_knee_ankle_angles):
        feedback.append("Close you knees more")
        return [0, feedback]

    if any(x < HIP_KNEE_ANKLE_MIN for x in hip_knee_ankle_angles):
        feedback.append("Open you knees more")
        return [0, feedback]

    if all(x > RETURN_THRESHOLD for x in shoulder_hip_knee_angles):
        return [0, []]

    if min_angle > MIN_THRESHOLD:
        feedback.append("Make sure to get closer to your knees")
        return [0, feedback]

    feedback.append("You're doing a situp")

    filename = str(math.floor(random.random() * 2**32))
    reps = min(count_line_intersections(shoulder_hip_knee_angles, MIN_THRESHOLD, filename="squat-min-" + filename + ".png"),
               count_line_intersections(shoulder_hip_knee_angles, RETURN_THRESHOLD, filename="squat-return-" + filename + ".png")) / 2

    return [reps, feedback]

'''

def asses_form_of_bridges(keypoints):
    print("asses_form_of_bridges called")
    
    [shoulder_hip_knee_angles,
     hip_knee_ankle_angles, hip_cords] = extract_situp_angles(keypoints)
    MIN_THRESHOLD = 12
    RETURN_THRESHOLD = 25
    HIP_KNEE_ANKLE_MIN = 35
    HIP_KNEE_ANKLE_MAX = 65
    feedback = []
    reps = 0
    min_angle = min(shoulder_hip_knee_angles)

    
    

    
    

    if all(x > RETURN_THRESHOLD for x in shoulder_hip_knee_angles):
        return [0, []]

    feedback.append("You're doing a bridge")

    filename = str(math.floor(random.random() * 2**32))
    reps = min(count_line_intersections(shoulder_hip_knee_angles, MIN_THRESHOLD, filename="bridges-min-" + filename + ".png"),
               count_line_intersections(shoulder_hip_knee_angles, RETURN_THRESHOLD, filename="bridges-return-" + filename + ".png")) / 2
    if reps > 0:
        first_return_angle_index = shoulder_hip_knee_angles.index(
            [item for item in shoulder_hip_knee_angles if item > RETURN_THRESHOLD][0])
        print("first_return_angle_index")
        print(hip_cords[first_return_angle_index])
        return [reps, [feedback, [str(x) for x in hip_cords[first_return_angle_index]]]]
    return [reps, [feedback, []]]


def asses_form_of_hip_abduction(keypoints):
    print("asses_form_of_hip_abductions called")
    [angles, xcords] = extract_situp_hip_abduction_from_keypoints(keypoints)
    MIN_THRESHOLD = 0.09
    RETURN_THRESHOLD = 0.13
    feedback = []
    reps = 0

    feedback.append("You're doing a hip abduction")

    filename = str(math.floor(random.random() * 2**32))
    reps = min(count_line_intersections(angles, MIN_THRESHOLD, filename=filename + "hip_abductions-min-" + ".png"),
               count_line_intersections(angles, RETURN_THRESHOLD, filename=filename + "hip_abductions-return-" + ".png")) / 2

    if reps > 0:
        first_return_angle_index = angles.index(
            [item for item in angles if item > RETURN_THRESHOLD][0])
        return [reps, [feedback, [str(angles[first_return_angle_index]), str(xcords[0])]]]
    return [reps, feedback, []]


def assess_form_of_JumpingJack(keypoints):
    """
    Assess jumping jack form based on ankle-knee-hip angles
    Args:
    keypoints: List of tensors where each tensor contains keypoint data for a frame.
    Returns:
    list: [number of reps, feedback list]
    """
    print("asses_form_of_JumpingJack called")
    filename = str(math.floor(random.random() * 2**32))
    angles = extract_base_ankles_angles(keypoints)

    LOWER_THRESHOLD = 55 
    UPPER_THRESHOLD = 70  
    feedback = []
    reps = 0
    
    reps1 = min(count_line_intersections(angles, LOWER_THRESHOLD,filename=filename + "-jj-min.png"),
               count_line_intersections(angles, UPPER_THRESHOLD,filename=filename + "-jj-max.png")) / 2
    
    reps2 = min(count_line_intersections(remove_anomalies_and_interpolate(angles), LOWER_THRESHOLD,filename=filename + "-jj-min.png"),
               count_line_intersections(remove_anomalies_and_interpolate(angles), UPPER_THRESHOLD,filename=filename + "-jj-max.png")) / 2  
    
    reps3 = min(count_line_intersections(gaussian_smoothing(angles), LOWER_THRESHOLD,filename=filename + "-jj-min.png"),
               count_line_intersections(gaussian_smoothing(angles), UPPER_THRESHOLD,filename=filename + "-jj-max.png")) / 2  

    if(reps1 == reps2 and reps1==reps3):
        reps = reps1 
    else :
        reps = reps3

    return [reps, feedback]


def assess_form_of_lunge(keypoints):
    """
    Assess lunge form based on ankle-knee-hip angles
    Args:
    keypoints: List of tensors where each tensor contains keypoint data for a frame.
    Returns:
    list: [number of reps, feedback list]
    """
    print("asses_form_of_lunge called")
    filename = str(math.floor(random.random() * 2**32))
    
    right_leg_angles , left_leg_angles = extract_lunges_angles(keypoints)
    highest_confidence_leg = determine_highest_confidence_for_lower(keypoints)
    angles = right_leg_angles if highest_confidence_leg == "right"  else left_leg_angles

    
    LOWER_THRESHOLD = 70  
    UPPER_THRESHOLD = 160  
    DANGER_THRESHOLD = 60  
    feedback = []
    reps = 0

    
    
    
    reps1 = min(count_line_intersections(angles, LOWER_THRESHOLD,filename=filename + "-lunge-min.png"),
               count_line_intersections(angles, UPPER_THRESHOLD,filename=filename + "-lunge-max.png")) / 2
    reps2 = min(count_line_intersections(remove_anomalies_and_interpolate(angles), LOWER_THRESHOLD,filename=filename + "-lunge-min.png"),
               count_line_intersections(remove_anomalies_and_interpolate(angles), UPPER_THRESHOLD,filename=filename + "-lunge-max.png")) / 2
    reps3 = min(count_line_intersections(gaussian_smoothing(angles), LOWER_THRESHOLD,filename=filename + "-lunge-min.png"),
               count_line_intersections(gaussian_smoothing(angles), UPPER_THRESHOLD,filename=filename + "-lunge-max.png")) / 2
    
    if(reps1 == reps2 and reps1==reps3):
        reps = reps1 
    else :
        reps = reps3
   
    return [reps, feedback]


def assess_form_of_pullup(keypoints):
    """
    Assess pull-up form based on shoulder-elbow-wrist angles
    Args:
    keypoints: List of tensors where each tensor contains keypoint data for a frame.    
    Returns:
    list: [number of reps, feedback list]
    """
    print("asses_form_of_lunge called")
    filename = str(math.floor(random.random() * 2**32))

    LOWER_THRESHOLD = 30  
    UPPER_THRESHOLD = 65  
    feedback = []

    right_arm_angles , left_arm_angles = extract_pullup_angles(keypoints)
    highest_confidence_arm = determine_highest_confidence_for_upper(keypoints)    
    angles = right_arm_angles if highest_confidence_arm == "right"  else left_arm_angles

    extra_angles , neg0 ,neg1  = extract_situp_angles(keypoints)

    for angle in angles :
        if angle > 45 :
            feedback.append("Warning: Elbows are not sufficiently flexed at the top of the pull-up.")

    if((sum(extra_angles) / len(extra_angles)) - 180 > 15) :
        feedback.append("Warning: Body is not straight; avoid swinging or arching the back.")        


    reps1 = min(count_line_intersections(angles, LOWER_THRESHOLD,filename=filename + "-pullup-min.png"),
               count_line_intersections(angles, UPPER_THRESHOLD,filename=filename + "-pullup-max.png")) / 2
    
    reps2 = min(count_line_intersections(remove_anomalies_and_interpolate(angles), LOWER_THRESHOLD,filename=filename + "-pullup-min.png"),
               count_line_intersections(remove_anomalies_and_interpolate(angles), UPPER_THRESHOLD,filename=filename + "-pullup-max.png")) / 2  
    
    reps3 = min(count_line_intersections(gaussian_smoothing(angles), LOWER_THRESHOLD,filename=filename + "-pullup-min.png"),
               count_line_intersections(gaussian_smoothing(angles), UPPER_THRESHOLD,filename=filename + "-pullup-max.png")) / 2  

    if(reps1 == reps2 and reps1==reps3):
        reps = reps1 
    else :
        reps = reps3

    return [reps, feedback]

