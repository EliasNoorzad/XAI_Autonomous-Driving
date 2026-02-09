# Preprocessing scripts (BDD100K → 640)

This repo uses three small scripts to prepare the dataset at **640×640**.

- **Train/Val letterbox to 640**  
  Resizes images and drivable masks to **640×640** using YOLO-style letterboxing (pad=114), and **updates YOLO boxes** accordingly. Outputs to `yolo_640/` and `drivable_masks_640/` for `train/` and `val/`.

- **Test letterbox to 640**  
  Same preprocessing as above, but for the `test/` split only. Outputs to `yolo_640/test/` and `drivable_masks_640/test/`.

- **Index builder for 640 dataset**  
  Scans `yolo_640/` + `drivable_masks_640/` and `daynight_labels.csv` and writes `index_640.csv` listing each image with available detection label, mask, and day/night tag.
