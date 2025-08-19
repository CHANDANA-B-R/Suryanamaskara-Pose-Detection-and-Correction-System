import cv2
import mediapipe as mp
import os
import numpy as np
import pickle

# --------------------------
# Path to your dataset
# Each pose should have its own folder inside this directory
# Example:
# dataset/
#   adho_mukh_svanasana/
#   ashtanga_namaskara/
#   ...
dataset_path = r"C:\Users\CHANDANA B R\OneDrive\Desktop\Project\Suryanamaskar"

# --------------------------
# MediaPipe setup
# --------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Dictionary to store keypoints for each pose
pose_keypoints = {}

# Iterate over each pose folder
for pose_name in os.listdir(dataset_path):
    pose_folder = os.path.join(dataset_path, pose_name)
    if not os.path.isdir(pose_folder):
        continue

    keypoints_list = []

    # Iterate over each image in the pose folder
    for img_file in os.listdir(pose_folder):
        img_path = os.path.join(pose_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            for lm in landmarks:
                keypoints.append([lm.x, lm.y])
            keypoints_list.append(np.array(keypoints).flatten())

    if keypoints_list:
        # Compute average keypoints for this pose
        avg_keypoints = np.mean(keypoints_list, axis=0)
        pose_keypoints[pose_name] = avg_keypoints.tolist()
        print(f"Computed average keypoints for {pose_name}")

# Save the reference keypoints to a file
with open("reference_keypoints.pkl", "wb") as f:
    pickle.dump(pose_keypoints, f)

print("\n✅ Reference keypoints saved as 'reference_keypoints.pkl'")
