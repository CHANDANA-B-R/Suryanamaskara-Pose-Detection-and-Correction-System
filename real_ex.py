# this is the code for detection only
import os
import cv2
import joblib
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp

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
]

# -------------------- Thresholds --------------------
CORRECTION_THRESHOLD = 0.25
HOLD_FRAMES = 15
CONSISTENT_FRAMES_REQUIRED = 5

# -------------------- State --------------------
current_pose_idx = 0
stable_ok_frames = 0
consistent_predicted_pose = None
consistent_count = 0

# -------------------- Helpers --------------------
def flatten_landmarks(results):
    if not results.pose_landmarks:
        return None
    return np.array(
        [val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y, lm.z, lm.visibility)],
        dtype=np.float32
    )

def normalize_landmarks(flat):
    arr = flat.copy()
    xs = arr[0::4]
    ys = arr[1::4]
    L_SH, R_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_HP, R_HP = mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value
    x_center = (xs[L_SH] + xs[R_SH] + xs[L_HP] + xs[R_HP]) / 4.0
    y_center = (ys[L_SH] + ys[R_SH] + ys[L_HP] + ys[R_HP]) / 4.0
    xs -= x_center
    ys -= y_center
    return arr

def _pt(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def compute_angles(results, frame_shape):
    """Compute key angles: elbows, knees, hips."""
    h, w = frame_shape[:2]
    lm = results.pose_landmarks.landmark

    pts = lambda part: _pt(lm[part], w, h)
    L_HIP, R_HIP = pts(mp_pose.PoseLandmark.LEFT_HIP.value), pts(mp_pose.PoseLandmark.RIGHT_HIP.value)
    L_KNEE, R_KNEE = pts(mp_pose.PoseLandmark.LEFT_KNEE.value), pts(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    L_ANK, R_ANK = pts(mp_pose.PoseLandmark.LEFT_ANKLE.value), pts(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    L_SH, R_SH = pts(mp_pose.PoseLandmark.LEFT_SHOULDER.value), pts(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    L_ELB, R_ELB = pts(mp_pose.PoseLandmark.LEFT_ELBOW.value), pts(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    L_WRS, R_WRS = pts(mp_pose.PoseLandmark.LEFT_WRIST.value), pts(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    return {
        "left_knee": angle_deg(L_HIP, L_KNEE, L_ANK),
        "right_knee": angle_deg(R_HIP, R_KNEE, R_ANK),
        "left_elbow": angle_deg(L_SH, L_ELB, L_WRS),
        "right_elbow": angle_deg(R_SH, R_ELB, R_WRS),
        "hip_level": (L_HIP[1] + R_HIP[1]) / 2.0,
        "shoulder_level": (L_SH[1] + R_SH[1]) / 2.0
    }

def plank_rule_override(angles):
    return (angles["left_knee"] >= 165 and angles["right_knee"] >= 165 and
            angles["left_elbow"] >= 150 and angles["right_elbow"] >= 150)

def ashtanga_rule_override(angles):
    elbows_ok = 80 <= angles["left_elbow"] <= 100 and 80 <= angles["right_elbow"] <= 100
    hips_low = angles["hip_level"] > angles["shoulder_level"]  # hips lower than shoulders
    knees_bent = angles["left_knee"] < 120 and angles["right_knee"] < 120
    return elbows_ok and hips_low and knees_bent

def draw_highlight_joints(image, results, joints):
    if not results.pose_landmarks:
        return
    h, w, _ = image.shape
    for joint in joints:
        enum_key = joint.upper()
        if enum_key in mp_pose.PoseLandmark.__members__:
            idx = mp_pose.PoseLandmark[enum_key].value
            lm = results.pose_landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)

def normalize_pose_name(name):
    return name.strip().lower().replace(" ", "_")

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

            angles = compute_angles(results, display.shape)

            # --- Rule-based overrides ---
            if target_pose_norm == "kumbhakasana" and plank_rule_override(angles):
                predicted_pose_norm = "kumbhakasana"
                predicted_pose = "kumbhakasana"

            if target_pose_norm == "ashtanga_namaskara" and ashtanga_rule_override(angles):
                predicted_pose_norm = "ashtanga_namaskara"
                predicted_pose = "ashtanga_namaskara"

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

            # --- Overlay info ---
            cv2.putText(display, f"Target Pose: {target_pose}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, f"Predicted: {predicted_pose}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(display, f"Holding... {stable_ok_frames}/{HOLD_FRAMES}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if stable_ok_frames >= HOLD_FRAMES:
                current_pose_idx += 1
                stable_ok_frames = 0
                consistent_predicted_pose = None
                consistent_count = 0
                cv2.putText(display, "Great! Next pose ▶", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Surya Namaskar", display)
                cv2.waitKey(700)

    cv2.imshow("Surya Namaskar", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

