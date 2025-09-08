# real_time_pose.py
import os
import cv2
import joblib
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp

# -------------------- Paths --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pose_classifier_rf.pkl")      # joblib dict: {"model", "label_encoder"}
REF_PATH   = os.path.join(HERE, "reference_keypoints.pkl")     # dict: {pose_name: [x,y,z]*33}

# -------------------- Load model & label encoder (with legacy fallback) --------------------
_loaded = joblib.load(MODEL_PATH)
if isinstance(_loaded, dict) and "model" in _loaded:
    model = _loaded["model"]
    label_encoder = _loaded.get("label_encoder", None)
else:
    # Legacy: model only (predicts string labels)
    model = _loaded
    label_encoder = None

# -------------------- Load reference keypoints (xyz per landmark) --------------------
with open(REF_PATH, "rb") as f:
    reference_keypoints = pickle.load(f)  # pose_name -> list length 99 (33*3)

# -------------------- Columns for model prediction (33 landmarks * [x,y,z,v]) --------------------
FEATURE_COLUMNS = [f"{i}_{c}" for i in range(1, 34) for c in ["x", "y", "z", "v"]]

# -------------------- MediaPipe setup --------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# -------------------- Pose order (sequence) --------------------
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

# -------------------- Thresholds & smoothing --------------------
CORRECTION_THRESHOLD = 0.12       # xy distance threshold in normalized coords
HOLD_FRAMES = 15                  # frames to hold correct pose
CONSISTENT_FRAMES_REQUIRED = 5    # smoothing to accept prediction

# -------------------- State --------------------
current_pose_idx = 0
stable_ok_frames = 0
consistent_predicted_pose = None
consistent_count = 0

# -------------------- Helpers --------------------
def normalize_pose_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def flatten_xyzw(results) -> np.ndarray:
    """Return 33*4 [x,y,z,visibility] as float32 for model prediction."""
    if not results.pose_landmarks:
        return None
    vals = []
    for lm in results.pose_landmarks.landmark:
        vals.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(vals, dtype=np.float32)

def extract_xyz(results) -> np.ndarray:
    """Return 33*3 [x,y,z] as float32 for correction/reference comparison."""
    if not results.pose_landmarks:
        return None
    vals = []
    for lm in results.pose_landmarks.landmark:
        vals.extend([lm.x, lm.y, lm.z])
    return np.array(vals, dtype=np.float32)

def center_xy_inplace(arr_xy_like: np.ndarray, step: int):
    """
    Center x and y by average of shoulders and hips.
    arr_xy_like: flattened array with step elements per landmark (e.g., step=4 for [x,y,z,v] or step=3 for [x,y,z]).
    Modifies in place and returns the same array.
    """
    xs = arr_xy_like[0::step]
    ys = arr_xy_like[1::step]
    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_HP = mp_pose.PoseLandmark.LEFT_HIP.value
    R_HP = mp_pose.PoseLandmark.RIGHT_HIP.value
    x_center = (xs[L_SH] + xs[R_SH] + xs[L_HP] + xs[R_HP]) / 4.0
    y_center = (ys[L_SH] + ys[R_SH] + ys[R_HP] + ys[R_HP]) / 4.0
    xs -= x_center
    ys -= y_center
    return arr_xy_like

def get_corrections(pose_name: str, cur_xyz_centered: np.ndarray):
    """
    Compare current centered xyz landmarks with reference (also xyz).
    Returns (corrections: list[str], highlights: list[str]).
    """
    corrections, highlights = [], []
    ref = reference_keypoints.get(pose_name, None)
    if ref is None:
        return corrections, highlights

    ref = np.array(ref, dtype=np.float32)
    if ref.size != 33 * 3 or cur_xyz_centered.size != 33 * 3:
        # mismatched reference or current vector
        return corrections, highlights

    # center reference (in case it was saved uncentered)
    ref_centered = center_xy_inplace(ref.copy(), 3)

    # compute xy deviations per joint
    JOINTS = {
        "left_shoulder":  mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        "left_elbow":     mp_pose.PoseLandmark.LEFT_ELBOW.value,
        "right_elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        "left_hip":       mp_pose.PoseLandmark.LEFT_HIP.value,
        "right_hip":      mp_pose.PoseLandmark.RIGHT_HIP.value,
        "left_knee":      mp_pose.PoseLandmark.LEFT_KNEE.value,
        "right_knee":     mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "left_ankle":     mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "right_ankle":    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    }

    for name, j in JOINTS.items():
        ix = j * 3
        dx = abs(cur_xyz_centered[ix + 0] - ref_centered[ix + 0])
        dy = abs(cur_xyz_centered[ix + 1] - ref_centered[ix + 1])
        # distance in xy (ignore z here to keep guidance stable)
        dist_xy = np.hypot(dx, dy)
        if dist_xy > CORRECTION_THRESHOLD:
            # Friendly text
            pretty = name.replace("_", " ")
            if "shoulder" in name or "elbow" in name or "wrist" in name:
                tip = f"Adjust your {pretty} alignment"
            elif "hip" in name:
                tip = f"Square your {pretty}"
            elif "knee" in name:
                tip = f"Align your {pretty} over ankle"
            else:
                tip = f"Adjust your {pretty}"
            corrections.append(tip)
            highlights.append(name)

    return corrections, highlights

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

# ---- Angle utilities (2D) for plank override ----
def _pt(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def compute_key_angles(results, frame_shape):
    h, w = frame_shape[:2]
    lm = results.pose_landmarks.landmark

    L_HIP  = _pt(lm[mp_pose.PoseLandmark.LEFT_HIP.value],  w, h)
    L_KNEE = _pt(lm[mp_pose.PoseLandmark.LEFT_KNEE.value], w, h)
    L_ANK  = _pt(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], w, h)

    R_HIP  = _pt(lm[mp_pose.PoseLandmark.RIGHT_HIP.value], w, h)
    R_KNEE = _pt(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value], w, h)
    R_ANK  = _pt(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value], w, h)

    L_SH   = _pt(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
    L_ELB  = _pt(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value],   w, h)
    L_WRS  = _pt(lm[mp_pose.PoseLandmark.LEFT_WRIST.value],  w, h)

    R_SH   = _pt(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
    R_ELB  = _pt(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value],   w, h)
    R_WRS  = _pt(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value],   w, h)

    return {
        "left_knee":  angle_deg(L_HIP, L_KNEE, L_ANK),
        "right_knee": angle_deg(R_HIP, R_KNEE, R_ANK),
        "left_elbow": angle_deg(L_SH,  L_ELB,  L_WRS),
        "right_elbow":angle_deg(R_SH,  R_ELB,  R_WRS),
    }

def plank_rule_override(results, frame_shape):
    """
    Returns True if landmarks strongly match 'plank-like':
    - both knees nearly straight (>= 165°)
    - elbows reasonably straight (>= 150°)
    """
    try:
        a = compute_key_angles(results, frame_shape)
        knees_straight  = (a["left_knee"]  >= 165.0) and (a["right_knee"] >= 165.0)
        elbows_straight = (a["left_elbow"] >= 150.0) and (a["right_elbow"] >= 150.0)
        return knees_straight and elbows_straight
    except Exception:
        return False

def decode_pred_label(pred):
    """Map prediction to string label (handles encoded ints and legacy models)."""
    if label_encoder is not None:
        # pred is an integer code
        return str(label_encoder.inverse_transform([int(pred)])[0])
    # legacy: model already outputs string
    return str(pred)

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

        # --- Prepare features for classifier ---
        flat_xyzw = flatten_xyzw(results)
        if flat_xyzw is None or flat_xyzw.size != 33 * 4:
            # reset smoothing if pose lost
            stable_ok_frames = 0
            consistent_predicted_pose = None
            consistent_count = 0
        else:
            X = pd.DataFrame([flat_xyzw], columns=FEATURE_COLUMNS)
            pred_raw = model.predict(X)[0]
            predicted_pose = decode_pred_label(pred_raw)
            predicted_pose_norm = normalize_pose_name(predicted_pose)

            # Optional rule-based override for Kumbhakasana (plank)
            if target_pose_norm == "kumbhakasana" and plank_rule_override(results, frame.shape):
                predicted_pose_norm = "kumbhakasana"
                predicted_pose = "kumbhakasana"

            # --- Corrections (use centered xyz vs reference xyz) ---
            cur_xyz = extract_xyz(results)
            cur_xyz_centered = center_xy_inplace(cur_xyz.copy(), 3) if cur_xyz is not None else None
            if cur_xyz_centered is None:
                corrections, highlights = [], []
            else:
                corrections, highlights = get_corrections(target_pose_norm, cur_xyz_centered)

            corrections_allowed = (len(corrections) == 0)

            # --- Smoothing ---
            if predicted_pose_norm == consistent_predicted_pose:
                consistent_count += 1
            else:
                consistent_predicted_pose = predicted_pose_norm
                consistent_count = 1

            if consistent_count >= CONSISTENT_FRAMES_REQUIRED:
                if predicted_pose_norm == target_pose_norm and corrections_allowed:
                    stable_ok_frames += 1
                else:
                    stable_ok_frames = 0
            else:
                stable_ok_frames = 0

            # --- Overlay texts ---
            cv2.putText(display, f"Target Pose: {target_pose}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, f"Predicted: {predicted_pose}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Corrections detected: {len(corrections)}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            draw_highlight_joints(display, results, highlights)

            y_base = 130
            for i, corr in enumerate(corrections[:5]):
                cv2.putText(display, corr, (10, y_base + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(display, f"Holding... {stable_ok_frames}/{HOLD_FRAMES}", (10, y_base + 5 * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            # --- Advance sequence when held correctly ---
            if stable_ok_frames >= HOLD_FRAMES:
                current_pose_idx += 1
                stable_ok_frames = 0
                consistent_predicted_pose = None
                consistent_count = 0
                cv2.putText(display, "Great! Next pose ▶", (10, y_base + 36 + 5 * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Surya Namaskar - Pose Detection & Correction", display)
                cv2.waitKey(700)
    else:
        # No pose detected -> reset smoothing
        stable_ok_frames = 0
        consistent_predicted_pose = None
        consistent_count = 0
        cv2.putText(display, "No pose detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # Footer: progress
    progress_text = f"Step {current_pose_idx+1}/{len(POSE_ORDER)}"
    cv2.putText(display, progress_text, (display.shape[1]-240, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow("Surya Namaskar - Pose Detection & Correction", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
