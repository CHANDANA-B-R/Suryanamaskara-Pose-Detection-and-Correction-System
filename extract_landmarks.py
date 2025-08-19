import os
import csv
import mediapipe as mp
import cv2

# ✅ Path to your organized dataset folder
BASE_PATH = r"C:\Users\CHANDANA B R\OneDrive\Desktop\Project\Suryanamaskar"

# Output CSV file
CSV_FILE = "pose_landmarks.csv"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image_path):
    """Extracts 33 pose landmarks from an image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not read image {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"⚠️ No pose detected in {image_path}")
        return None

    # Flatten all (x, y, z, visibility) values into a single list
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

    return landmarks

def main():
    header_written = False

    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)

        for split in ["train", "valid", "test"]:
            split_path = os.path.join(BASE_PATH, split)

            for pose_name in os.listdir(split_path):
                pose_folder = os.path.join(split_path, pose_name)

                if not os.path.isdir(pose_folder):
                    continue

                for img_name in os.listdir(pose_folder):
                    img_path = os.path.join(pose_folder, img_name)

                    landmarks = extract_landmarks(img_path)
                    if landmarks is None:
                        continue

                    # Write header (landmark1_x, landmark1_y, ... , label)
                    if not header_written:
                        num_coords = len(landmarks)
                        header = [f"{i}_{coord}" for i in range(1, 34) for coord in ["x", "y", "z", "v"]]
                        header.append("label")
                        writer.writerow(header)
                        header_written = True

                    # Write row: landmark values + label
                    writer.writerow(landmarks + [pose_name])

                    print(f"✅ Processed {img_name} → {pose_name}")

    print(f"\n🎯 Landmarks saved to {CSV_FILE}")

if __name__ == "__main__":
    main()
