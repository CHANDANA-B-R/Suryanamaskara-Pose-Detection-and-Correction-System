import cv2
import mediapipe as mp
import os
import numpy as np
import pickle

# --------------------------
# Path to your TRAIN dataset only
# --------------------------
DATASET_PATH = r"C:\Users\CHANDANA B R\OneDrive\Desktop\pro\Project\Suryanamaskar\train"

# Allowed image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --------------------------
# MediaPipe setup
# --------------------------
mp_pose = mp.solutions.pose

pose_keypoints = {}

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    for pose_name in os.listdir(DATASET_PATH):
        pose_folder = os.path.join(DATASET_PATH, pose_name)
        if not os.path.isdir(pose_folder):
            continue

        keypoints_list = []

        for img_file in os.listdir(pose_folder):
            if not img_file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(pose_folder, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Could not read {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                for lm in landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z])  # Include z-axis
                keypoints_list.append(np.array(keypoints).flatten())

        if keypoints_list:
            avg_keypoints = np.mean(keypoints_list, axis=0)
            pose_keypoints[pose_name] = avg_keypoints.tolist()
            print(f"✅ Computed average keypoints for: {pose_name}")
        else:
            print(f"⚠️ No valid keypoints found for: {pose_name}")

# Save the reference keypoints to a file
with open("reference_keypoints.pkl", "wb") as f:
    pickle.dump(pose_keypoints, f)

print("\n🎯 Reference keypoints saved as 'reference_keypoints.pkl'")
