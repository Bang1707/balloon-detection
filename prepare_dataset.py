import os
import csv
import ast
import random
import shutil

# ---------------- CONFIG ----------------
CSV_PATH = "balloon-data.csv"
IMAGES_DIR = "images"          # kaggle'dan gelen images/
OUTPUT_DIR = "data/balloon"    # hedef
TRAIN_RATIO = 0.8
CLASS_ID = 0                   # balloon
# ----------------------------------------

random.seed(42)

os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)

# CSV oku
with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = list(csv.DictReader(f))

random.shuffle(reader)

split_idx = int(len(reader) * TRAIN_RATIO)
train_rows = reader[:split_idx]
val_rows = reader[split_idx:]


def process(rows, split):
    for row in rows:
        fname = row["fname"]
        img_w = int(row["width"])
        img_h = int(row["height"])
        bboxes = ast.literal_eval(row["bbox"])

        # image kopyala
        src_img = os.path.join(IMAGES_DIR, fname)
        dst_img = os.path.join(OUTPUT_DIR, "images", split, fname)
        shutil.copy(src_img, dst_img)

        label_path = os.path.join(
            OUTPUT_DIR, "labels", split, fname.replace(".jpg", ".txt")
        )

        with open(label_path, "w") as f:
            for box in bboxes:
                xmin = box["xmin"]
                ymin = box["ymin"]
                xmax = box["xmax"]
                ymax = box["ymax"]

                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h

                f.write(
                    f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )


process(train_rows, "train")
process(val_rows, "val")

print("✅ Dataset hazır: YOLO formatına dönüştürüldü.")
