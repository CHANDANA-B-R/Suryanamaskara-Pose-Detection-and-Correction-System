# this is the code for detection and correction system
import os
import cv2
import joblib
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
import time

# -------------------- Paths --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pose_classifier_rf.pkl")
REF_PATH   = os.path.join(HERE, "reference_keypoints.pkl")

# -------------------- Load model & refs --------------------
model = joblib.load(MODEL_PATH)
with open(REF_PATH, "rb") as f:
    reference_keypoints = pickle.load(f)

FEATURE_COLUMNS = [f"{i}_{c}" for i in range(1, 34) for c in ["x", "y", "z", "v"]]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# -------------------- Pose order --------------------
POSE_ORDER = [
    "pranamasana",
    "hasta_utthanasana",
    "padahastasana",
    "ashwa_sanchalanasana",
    "kumbhakasana",
    "ashtanga_namaskara",
    "bhujangasana",
    "adho_mukh_svanasana",
    "ashwa_sanchalanasana",
    "padahastasana",
    "hasta_utthanasana",
    "pranamasana",
]

# -------------------- Correction Rules --------------------
POSE_CORRECTIONS = {
    "pranamasana": {
        "elbow_angle": (170, 190, "Keep arms straight together"),
    },
    "hasta_utthanasana": {
        "elbow_angle": (170, 190, "Arms straight up"),
        "back_angle": (190, 230, "Arch back slightly"),
    },
    "padahastasana": {
        "hip_angle": (50, 100, "Bend forward fully"),
    },
    "ashwa_sanchalanasana": {
        "front_knee": (80, 100, "Bend front knee to ~90°"),
        "back_leg": (160, 190, "Keep back leg straight"),
    },
    "kumbhakasana": {
        "body_line": (160, 180, "Keep body straight like plank"),
    },
    "ashtanga_namaskara": {
        "elbow_angle": (80, 110, "Bend elbows ~90°"),
    },
    "bhujangasana": {
        "back_angle": (90, 120, "Lift chest higher"),
        "elbow_angle": (160, 190, "Keep arms straight"),
    },
    "adho_mukh_svanasana": {
        "hip_angle": (70, 110, "Push hips up to form inverted V"),
    },
}

# -------------------- Helpers --------------------
def flatten_landmarks(results):
    if not results.pose_landmarks:
        return None
    return np.array(
        [val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y, lm.z, lm.visibility)],
        dtype=np.float32
    )

def normalize_pose_name(name):
    return name.strip().lower().replace(" ", "_")

def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def get_point(lm, idx, w, h):
    return [lm[idx].x * w, lm[idx].y * h]

def check_corrections(pose_name, lm, frame_shape):
    """Return list of feedback messages for wrong alignment"""
    if pose_name not in POSE_CORRECTIONS:
        return []
    h, w, _ = frame_shape
    feedback = []

    for check, (low, high, message) in POSE_CORRECTIONS[pose_name].items():
        angle = None
        if check == "elbow_angle":
            angle = calculate_angle(get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_ELBOW.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h))
        elif check == "front_knee":
            angle = calculate_angle(get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h))
        elif check == "hip_angle":
            angle = calculate_angle(get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h))
        elif check == "back_angle":
            angle = calculate_angle(get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h))
        elif check == "body_line":
            angle = calculate_angle(get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h))
        elif check == "back_leg":
            angle = calculate_angle(get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h),
                                    get_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h))
        if angle is not None and not (low <= angle <= high):
            feedback.append(message)

    return feedback

# -------------------- State --------------------
current_pose_idx = 0
stable_ok_frames = 0
consistent_predicted_pose = None
consistent_count = 0
CONSISTENT_FRAMES_REQUIRED = 5
HOLD_FRAMES = 15

# -------------------- Main loop --------------------
cap = cv2.VideoCapture(0)

while cap.isOpened() and current_pose_idx < len(POSE_ORDER):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    display = frame.copy()
    target_pose = POSE_ORDER[current_pose_idx]
    target_pose_norm = normalize_pose_name(target_pose)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        flat = flatten_landmarks(results)

        if flat is not None:
            X = pd.DataFrame([flat], columns=FEATURE_COLUMNS)
            predicted_pose = model.predict(X)[0]
            predicted_pose_norm = normalize_pose_name(predicted_pose)

            # --- Smoothing logic ---
            if predicted_pose_norm == consistent_predicted_pose:
                consistent_count += 1
            else:
                consistent_predicted_pose = predicted_pose_norm
                consistent_count = 1

            if consistent_count >= CONSISTENT_FRAMES_REQUIRED and predicted_pose_norm == target_pose_norm:
                stable_ok_frames += 1
            else:
                stable_ok_frames = 0

            # --- Correction Feedback ---
            feedback = check_corrections(predicted_pose_norm, results.pose_landmarks.landmark, display.shape)

            # --- Overlay info ---
            cv2.putText(display, f"Target Pose: {target_pose}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, f"Predicted: {predicted_pose}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(display, f"Holding... {stable_ok_frames}/{HOLD_FRAMES}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if feedback:
                for i, msg in enumerate(feedback):
                    cv2.putText(display, msg, (10, 150 + i*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif predicted_pose_norm == target_pose_norm:
                cv2.putText(display, "✔ Good Alignment", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

            if stable_ok_frames >= HOLD_FRAMES:
                current_pose_idx += 1
                stable_ok_frames = 0
                consistent_predicted_pose = None
                consistent_count = 0
                cv2.putText(display, "Great! Next pose ▶", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Surya Namaskar", display)
                cv2.waitKey(700)

    cv2.imshow("Surya Namaskar", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
