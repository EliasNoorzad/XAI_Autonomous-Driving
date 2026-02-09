# Models (Weights)

These files are trained weights for our BDD100K (640×640) experiments.

## Files
- `yolov8n_det_best.pt`  
  Baseline object detection model (YOLOv8n) trained on 5 classes: person, car, bike, bus, truck.

- `yolov8n_det+seg_best.pt`  
  Multi-task model: detection + drivable-area segmentation.

- `yolov8n_det+seg_cbam_best.pt`  
  Multi-task model: detection + drivable-area segmentation + CBAM.

- `yolov8n_tritask_cbam_best.pt`  
  Tri-task model with CBAM attention integrated in the architecture.

## Notes
- All models use input resolution **640×640**.
- These weights are used by the scripts/notebooks in the `evaluation/` folder.
