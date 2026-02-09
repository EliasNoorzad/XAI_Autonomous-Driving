
import os
import csv

DATASET_ROOT = r"C:\Users\alias\Documents\Polito\XAI\Project\Dataset"

#  using the preprocessed 640 folders
YOLO_ROOT = os.path.join(DATASET_ROOT, "yolo_640")
MASK_ROOT = os.path.join(DATASET_ROOT, "drivable_masks_640")

TAG_CSV = os.path.join(DATASET_ROOT, "daynight_labels.csv")

# writing a separate index for 640 
OUT_INDEX = os.path.join(DATASET_ROOT, "index_640.csv")

SPLITS = ["train", "val", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def rel(path: str) -> str:
    return os.path.relpath(path, DATASET_ROOT).replace("\\", "/")


def load_daynight_tags():
    tags = {}
    if not os.path.exists(TAG_CSV):
        return tags

    with open(TAG_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip()
            image_id = row["image_id"].strip()
            label = row["label"].strip()
            if label in ("day", "night"):
                tags[(split, image_id)] = label
    return tags


def build_index():
    tags = load_daynight_tags()
    rows = []

    for split in SPLITS:
        img_dir = os.path.join(YOLO_ROOT, split, "images")
        det_dir = os.path.join(YOLO_ROOT, split, "labels")
        mask_dir = os.path.join(MASK_ROOT, split)

        if not os.path.isdir(img_dir):
            continue

        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(IMG_EXTS):
                continue

            image_id = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)

            det_path = os.path.join(det_dir, image_id + ".txt")
            has_det = 1 if os.path.exists(det_path) else 0

            mask_path = os.path.join(mask_dir, image_id + ".png")
            has_mask = 1 if os.path.exists(mask_path) else 0

            tag = tags.get((split, image_id), "")
            has_tag = 1 if tag in ("day", "night") else 0

            rows.append({
                "split": split,
                "image_id": image_id,
                "image_rel": rel(img_path),
                "det_rel": rel(det_path) if has_det else "",
                "mask_rel": rel(mask_path) if has_mask else "",
                "tag": tag if has_tag else "",
                "has_det": has_det,
                "has_mask": has_mask,
                "has_tag": has_tag,
            })

    with open(OUT_INDEX, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "image_id", "image_rel", "det_rel", "mask_rel", "tag",
                      "has_det", "has_mask", "has_tag"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: wrote {OUT_INDEX}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    build_index()

