import os
from pathlib import Path
from PIL import Image
import numpy as np
IMGSZ = 640

def read_yolo_labels(label_path: Path):
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            rows.append([float(cls), float(x), float(y), float(w), float(h)])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(rows, dtype=np.float32)

def write_yolo_labels(label_path: Path, labels: np.ndarray):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for r in labels:
            f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")

def letterbox_image_and_mask(img: Image.Image, mask: Image.Image, new=IMGSZ):
    W0, H0 = img.size
    r = min(new / W0, new / H0)
    nw, nh = int(round(W0 * r)), int(round(H0 * r))

    pad_w = new - nw
    pad_h = new - nh
    left = pad_w // 2
    top  = pad_h // 2

    img_r  = img.resize((nw, nh), resample=Image.BILINEAR)
    mask_r = mask.resize((nw, nh), resample=Image.NEAREST)

    img_lb = Image.new("RGB", (new, new), (114, 114, 114))
    img_lb.paste(img_r, (left, top))

    mask_lb = Image.new("L", (new, new), 0)
    mask_lb.paste(mask_r, (left, top))

    return img_lb, mask_lb, (W0, H0, r, left, top)

def letterbox_labels(labels: np.ndarray, meta, new=IMGSZ):
    if labels.size == 0:
        return labels
    W0, H0, r, left, top = meta

    out = labels.copy()
    xywh = out[:, 1:5]

    x = xywh[:, 0] * W0
    y = xywh[:, 1] * H0
    w = xywh[:, 2] * W0
    h = xywh[:, 3] * H0

    x = x * r + left
    y = y * r + top
    w = w * r
    h = h * r

    xywh[:, 0] = x / new
    xywh[:, 1] = y / new
    xywh[:, 2] = w / new
    xywh[:, 3] = h / new

    out[:, 1:5] = np.clip(xywh, 0.0, 1.0)
    return out

def process_split(yolo_root, mask_root, out_yolo_root, out_mask_root, split):
    yolo_root = Path(yolo_root)
    mask_root = Path(mask_root)
    out_yolo_root = Path(out_yolo_root)
    out_mask_root = Path(out_mask_root)

    img_dir = yolo_root / split / "images"
    lbl_dir = yolo_root / split / "labels"
    msk_dir = mask_root / split

    out_img_dir = out_yolo_root / split / "images"
    out_lbl_dir = out_yolo_root / split / "labels"
    out_msk_dir = out_mask_root / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    img_paths = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    img_paths.sort()

    total = len(img_paths)
    for i, img_path in enumerate(img_paths, 1):
        if i % 200 == 0 or i == total:
            print(f"{split}: {i}/{total}", flush=True)

        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        msk_path = msk_dir / f"{stem}.png"

        if not msk_path.exists():
            raise FileNotFoundError(f"Missing mask: {msk_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")
        labels = read_yolo_labels(lbl_path)

        img_lb, mask_lb, meta = letterbox_image_and_mask(img, mask, new=IMGSZ)
        labels_lb = letterbox_labels(labels, meta, new=IMGSZ)

        mask_np = (np.array(mask_lb, dtype=np.uint8) > 0).astype(np.uint8) * 255
        mask_lb = Image.fromarray(mask_np, mode="L")

        out_img_path = out_img_dir / f"{stem}.jpg"
        out_lbl_path = out_lbl_dir / f"{stem}.txt"
        out_msk_path = out_msk_dir / f"{stem}.png"

        img_lb.save(out_img_path, quality=95)
        mask_lb.save(out_msk_path)
        write_yolo_labels(out_lbl_path, labels_lb)

def main():
    YOLO_ROOT = r"C:\Users\alias\Documents\Polito\XAI\Project\Dataset\yolo"
    MASK_ROOT = r"C:\Users\alias\Documents\Polito\XAI\Project\Dataset\drivable_masks"

    OUT_YOLO_ROOT = r"C:\Users\alias\Documents\Polito\XAI\Project\Dataset\yolo_640"
    OUT_MASK_ROOT = r"C:\Users\alias\Documents\Polito\XAI\Project\Dataset\drivable_masks_640"

    for split in ["train", "val"]:
        process_split(YOLO_ROOT, MASK_ROOT, OUT_YOLO_ROOT, OUT_MASK_ROOT, split)

    print("Done.")

if __name__ == "__main__":
    main()
