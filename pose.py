# real_time_pose.py
"""
Real-time Surya Namaskar POSE DETECTION (sequence only)

- Loads saved model (expects joblib file that contains {"model":..., "label_encoder":...}).
- Uses MediaPipe to detect pose landmarks.
- Predicts pose using the trained classifier.
- Advances through POSE_ORDER when the predicted pose matches the target consistently.
- Press 'q' to quit.
"""

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from time import time

# -------------------- Paths --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pose_classifier_rf.pkl")

# -------------------- Pose order (target sequence) --------------------
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

# -------------------- Smoothing / thresholds --------------------
CONSISTENT_FRAMES_REQUIRED = 5   # number of consecutive frames the same pose must be predicted
HOLD_FRAMES_BEFORE_ADVANCE = 10  # additional holding frames to give the user (optional visible count)

# -------------------- Helpers --------------------
def flatten_landmarks(results):
    """Return flattened list of (x,y,z,visibility) for 33 landmarks or None if no landmarks."""
    if not results.pose_landmarks:
        return None
    return np.array(
        [val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y, lm.z, lm.visibility)],
        dtype=np.float32
    )

def normalize_pose_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

# -------------------- Load model & label encoder --------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

loaded = joblib.load(MODEL_PATH)

# Support both single-object and dict-with-encoder saves
if isinstance(loaded, dict):
    model = loaded.get("model") or loaded.get("estimator") or loaded
    label_encoder = loaded.get("label_encoder", None)
else:
    model = loaded
    label_encoder = None

if model is None:
    raise ValueError("Could not find model in the loaded file.")

# If label_encoder exists and was stored as sklearn LabelEncoder, we will use it to decode numeric preds
def decode_label(pred):
    if label_encoder is None:
        return str(pred)
    try:
        return str(label_encoder.inverse_transform([int(pred)])[0])
    except Exception:
        # If model already returns string labels
        return str(pred)

# Feature column names expected (33 landmarks × 4 values each)
FEATURE_COLUMNS = [f"{i}_{c}" for i in range(1, 34) for c in ["x", "y", "z", "v"]]

# -------------------- MediaPipe setup --------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# We'll use a context manager to ensure resources are freed on exit
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible. Make sure a camera is connected and not used by another app.")

    # state for smoothing / sequence progression
    current_pose_idx = 0
    consistent_predicted_pose = None
    consistent_count = 0
    hold_count = 0

    fps_last_time = time()
    fps_frame_count = 0
    fps = 0.0

    window_name = "Surya Namaskar - Sequence Detection (press 'q' to quit)"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        display = frame.copy()

        # compute FPS (simple)
        fps_frame_count += 1
        if time() - fps_last_time >= 1.0:
            fps = fps_frame_count / (time() - fps_last_time)
            fps_last_time = time()
            fps_frame_count = 0

        target_pose_raw = POSE_ORDER[current_pose_idx]
        target_pose = normalize_pose_name(target_pose_raw)

        pred_label = "None"
        predicted_pose_norm = None

        flat = flatten_landmarks(results)
        if flat is not None:
            # ensure correct shape: 132 features (33*4)
            if flat.size != 33 * 4:
                # corrupted / unexpected landmark length
                cv2.putText(display, "Unexpected landmark size", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                # create dataframe with correct column ordering (same as training)
                X = pd.DataFrame([flat], columns=FEATURE_COLUMNS)

                # predict (model may expect numeric features same as training)
                try:
                    y_pred = model.predict(X)[0]
                    pred_label = decode_label(y_pred)
                except Exception as e:
                    # fallback: try passing raw numpy
                    try:
                        y_pred = model.predict([flat])[0]
                        pred_label = decode_label(y_pred)
                    except Exception as e2:
                        pred_label = f"err"
                        print("Prediction error:", e, e2)

                predicted_pose_norm = normalize_pose_name(pred_label)

                # smoothing: require same pose for N consecutive frames
                if predicted_pose_norm == consistent_predicted_pose:
                    consistent_count += 1
                else:
                    consistent_predicted_pose = predicted_pose_norm
                    consistent_count = 1
                    hold_count = 0

                # When we have a consistent prediction:
                if consistent_count >= CONSISTENT_FRAMES_REQUIRED:
                    if predicted_pose_norm == target_pose:
                        # increment hold counter (small visible buffer before advancing)
                        hold_count += 1
                        cv2.putText(display, f"Holding: {hold_count}/{HOLD_FRAMES_BEFORE_ADVANCE}", (10, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        if hold_count >= HOLD_FRAMES_BEFORE_ADVANCE:
                            # advance sequence
                            current_pose_idx += 1
                            hold_count = 0
                            consistent_count = 0
                            consistent_predicted_pose = None

                            # if reached end, show success and break loop after short pause
                            if current_pose_idx >= len(POSE_ORDER):
                                cv2.putText(display, "🎉 Sequence complete!", (10, 170),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
                                cv2.imshow(window_name, display)
                                cv2.waitKey(1500)
                                break
                    else:
                        # predicted consistently but not equal to target
                        hold_count = 0

        else:
            # no landmarks detected — reset smoothing
            consistent_predicted_pose = None
            consistent_count = 0
            hold_count = 0
            cv2.putText(display, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Draw landmarks if present
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Overlays: target, predicted, progress, fps
        cv2.putText(display, f"Target: {target_pose_raw}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(display, f"Predicted: {pred_label}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(display, f"Step: {current_pose_idx+1}/{len(POSE_ORDER)}", (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
