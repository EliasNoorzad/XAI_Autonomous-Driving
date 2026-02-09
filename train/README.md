# Training Notebooks

This folder contains the Colab notebooks used to train our models on BDD100K at 640×640 resolution.

## Notebooks
- `Baseline_Object_Detection.ipynb`  
  Trains the baseline YOLOv8 object detector (no segmentation/tagging).

- `det+seg_(no_attention).ipynb`  
  Trains the multi-task baseline without attention: **detection + drivable-area segmentation**.

- `Seg+Det_(with_attention).ipynb`  
  Trains **detection + drivable-area segmentation** with **CBAM attention** integrated into the architecture.

- `Det+Seg+tag__(with_attention).ipynb`  
  Trains the full **tri-task** model with **CBAM attention**: **detection + drivable-area segmentation + day/night tagging**.

## Notes
- All notebooks use a fixed input size of **640×640**.
- Produced weights are stored in the `models/` folder.
