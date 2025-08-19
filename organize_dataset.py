import os
import json
import shutil

base_dir = r"C:\Users\CHANDANA B R\OneDrive\Desktop\Project\Suryanamaskar"
splits = ["train", "valid", "test"]

for split in splits:
    split_path = os.path.join(base_dir, split)
    ann_path = os.path.join(split_path, "_annotations.coco.json")

    if not os.path.exists(ann_path):
        print(f"⚠️ No annotation file found for {split}, skipping...")
        continue

    print(f"\n📂 Processing {split} dataset...")

    with open(ann_path, 'r') as f:
        data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in data['categories']}

    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']

        image_info = next(img for img in data['images'] if img['id'] == image_id)
        filename = image_info['file_name']

        class_name = categories[category_id]
        class_folder = os.path.join(split_path, class_name)
        os.makedirs(class_folder, exist_ok=True)

        src = os.path.join(split_path, filename)

        # Shorten filename to avoid Windows MAX_PATH issue
        new_filename = f"{class_name}_{image_id}.jpg"
        dst = os.path.join(class_folder, new_filename)

        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✅ Moved: {filename} → {new_filename}")
        else:
            print(f"⚠️ Image not found: {filename}")

print("\n🎯 Dataset organization completed!")
