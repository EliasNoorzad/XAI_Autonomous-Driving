# BDD100K preprocessing (640×640)

These 3 scripts generate the **640×640** version of our dataset and a simple CSV index.

## 1) Train/Val → 640 (letterbox + fix labels + masks)
**What it does**
- Reads:  
  `yolo/{train,val}/images/`, `yolo/{train,val}/labels/`, `drivable_masks/{train,val}/`
- Applies **YOLO-style letterbox** resize to **640×640** (keeps aspect ratio, pads with **114**).
- Updates YOLO boxes to match the resized + padded image.
- Writes:
  - `yolo_640/{train,val}/images/`
  - `yolo_640/{train,val}/labels/`
  - `drivable_masks_640/{train,val}/` (binary masks saved as 0/255 PNG)

## 2) Test → 640 (same, test split only)
**What it does**
- Same pipeline as (1), but only for the `test/` split.
- Writes:
  - `yolo_640/test/images/`
  - `yolo_640/test/labels/`
  - `drivable_masks_640/test/`

## 3) Build `index_640.csv`
**What it does**
- Scans `yolo_640/{train,val,test}/images/`
- Checks if each sample has:
  - detection label file (`yolo_640/.../labels/<id>.txt`)
  - drivable mask (`drivable_masks_640/.../<id>.png`)
  - day/night tag from `daynight_labels.csv`
- Writes `index_640.csv` with relative paths + flags (`has_det`, `has_mask`, `has_tag`).

## Output folders
After running, you should have:
- `yolo_640/` (images + corrected YOLO labels)
- `drivable_masks_640/` (640×640 binary masks)
- `index_640.csv`
