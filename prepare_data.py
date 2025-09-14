# prepare_dataset.py
import os
import shutil
import pandas as pd

# -- USER PATHS (relative to project root) --
DATA_DIR = "data"
IMG_DIR1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
META_CSV = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
OUT_DIR = os.path.join(DATA_DIR, "processed")   # <-- this will be created

# Create output folder
os.makedirs(OUT_DIR, exist_ok=True)

# Read metadata CSV
print("Loading metadata:", META_CSV)
df = pd.read_csv(META_CSV)

# Ensure we have the expected columns
if "image_id" not in df.columns or "dx" not in df.columns:
    raise SystemExit("metadata CSV must contain 'image_id' and 'dx' columns.")

# Copy images into class subfolders (no train/val split here)
missing = []
count = 0
for _, row in df.iterrows():
    img_name = str(row["image_id"]) + ".jpg"
    label = str(row["dx"])
    # destination dir: data/processed/<label>/
    dst_dir = os.path.join(OUT_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)

    # try both image folders
    src = os.path.join(IMG_DIR1, img_name)
    if not os.path.exists(src):
        src = os.path.join(IMG_DIR2, img_name)

    if os.path.exists(src):
        dst = os.path.join(dst_dir, img_name)
        # don't copy if already exists (safe to re-run)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        count += 1
    else:
        missing.append(img_name)

# Summary
print(f"Copied {count} images into '{OUT_DIR}' (class subfolders).")
if missing:
    print(f"Warning: {len(missing)} images were not found. Example missing: {missing[:5]}")
else:
    print("All images found and copied.")
