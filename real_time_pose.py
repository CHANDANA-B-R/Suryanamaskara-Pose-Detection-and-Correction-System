import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle
import pyttsx3
import time

# --------------------------
# Load trained Random Forest model
# --------------------------
model = joblib.load("pose_classifier_rf.pkl")

# --------------------------
# Load reference keypoints
# --------------------------
with open('reference_keypoints.pkl', 'rb') as f:
    reference_keypoints = pickle.load(f)

# --------------------------
# MediaPipe Pose setup
# --------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --------------------------
# Audio engine setup
# --------------------------
engine = pyttsx3.init()

# --------------------------
# Pose order for Surya Namaskar
# --------------------------
pose_order = [
    "pranamasana",
    "hasta_utthanasana",
    "ashwa_sanchalanasana",
    "bhujangasana",
    "adho_mukh_svanasana",
    "kumbhakasana",
    "padahastasana",
    "ashtanga_namaskara"
]

current_pose_index = 0
last_suggestions = {}
pose_completed = False
feedback_cooldown = 1.5  # seconds between feedback

# --------------------------
# Helper functions
# --------------------------
def extract_keypoints(results):
    """Extract 132 features: x, y, z, visibility for 33 landmarks"""
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints)

def normalize_keypoints(keypoints):
    """Normalize keypoints relative to torso center (shoulders & hips)"""
    keypoints = keypoints.copy()
    x_coords = keypoints[0::4]
    y_coords = keypoints[1::4]

    left_shoulder = x_coords[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = x_coords[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = x_coords[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = x_coords[mp_pose.PoseLandmark.RIGHT_HIP.value]

    x_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
    y_center = (y_coords[mp_pose.PoseLandmark.LEFT_SHOULDER.value] +
                y_coords[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] +
                y_coords[mp_pose.PoseLandmark.LEFT_HIP.value] +
                y_coords[mp_pose.PoseLandmark.RIGHT_HIP.value]) / 4

    keypoints[0::4] -= x_center
    keypoints[1::4] -= y_center
    return keypoints

def get_corrections(pose_name, keypoints, threshold=0.05):
    """Compare user keypoints with reference and return corrections + highlight joints"""
    global last_suggestions
    corrections = []
    highlight_joints = []

    reference = reference_keypoints.get(pose_name)
    if reference is None:
        return corrections, highlight_joints

    deviations = np.abs(keypoints - reference)

    # Joints to check
    joints = {
        'left_elbow': 13*4,
        'right_elbow': 14*4,
        'left_knee': 25*4,
        'right_knee': 26*4,
        'left_shoulder': 11*4,
        'right_shoulder': 12*4
    }

    for joint_name, idx in joints.items():
        if deviations[idx] > threshold or deviations[idx+1] > threshold:
            suggestion = f"Adjust your {joint_name.replace('_', ' ')}"
            if last_suggestions.get(joint_name) != suggestion:
                corrections.append(suggestion)
                last_suggestions[joint_name] = suggestion
            highlight_joints.append(joint_name)

    # Remove joints that are aligned now
    for joint_name in list(last_suggestions.keys()):
        idx = joints[joint_name]
        if deviations[idx] <= threshold and deviations[idx+1] <= threshold:
            last_suggestions.pop(joint_name)

    return corrections, highlight_joints

def is_pose_correct(corrections):
    """Pose is correct if no corrections"""
    return len(corrections) == 0

# --------------------------
# Start webcam feed
# --------------------------
cap = cv2.VideoCapture(0)
last_feedback_time = time.time()

while cap.isOpened() and current_pose_index < len(pose_order):
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        keypoints = extract_keypoints(results)
        keypoints = normalize_keypoints(keypoints)
        keypoints_reshaped = keypoints.reshape(1, -1)

        # Predict pose
        predicted_pose = model.predict(keypoints_reshaped)[0]
        target_pose = pose_order[current_pose_index]

        # Only evaluate current target pose
        corrections, highlight_joints = get_corrections(target_pose, keypoints)

        # Display current target pose
        cv2.putText(image, f'Do this pose: {target_pose}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Visual feedback on joints
        for joint_name in highlight_joints:
            idx = mp_pose.PoseLandmark[joint_name.upper()].value
            lm = results.pose_landmarks.landmark[idx]
            h, w, _ = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)

        # Provide feedback if pose is not correct
        if not is_pose_correct(corrections):
            # Limit feedback frequency
            if time.time() - last_feedback_time > feedback_cooldown:
                for c in corrections:
                    cv2.putText(image, c, (10, 70 + 30 * corrections.index(c)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    engine.say(c)
                engine.runAndWait()
                last_feedback_time = time.time()
        else:
            # Pose done correctly → move to next
            current_pose_index += 1
            last_suggestions = {}
            cv2.putText(image, "Good! Move to next pose.", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            engine.say("Good! Move to next pose.")
            engine.runAndWait()
            time.sleep(1)  # small pause before next pose

    cv2.imshow('Surya Namaskar Guided Pose System', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
