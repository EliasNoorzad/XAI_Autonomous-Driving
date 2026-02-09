# Evaluation (Test Set)

This folder contains the notebooks used to evaluate the trained models on the BDD100K **test set** and generate the results included in the report.

## Notebooks
- `01_baseline_det_test.ipynb`  
  Evaluates the **baseline detection** model on the **test set** and reports detection metrics.

- `02_baseline_det_seg_test.ipynb`  
  Evaluates the **detection + drivable-area segmentation** baseline (no attention) on the **test set** and reports detection/segmentation metrics.

- `03_tritask_cbam_test.ipynb`  
  Evaluates the **tri-task model with CBAM attention** (detection + segmentation + day/night) on the **test set** and reports tri-task performance.

- `04_perturbation.ipynb`  
  Runs the **occlusion/perturbation faithfulness test** using the learned attention map and measures performance drops under occlusion.

- `05_overlays.ipynb`  
  Generates **qualitative visualizations** (attention overlays and prediction overlays) for selected test images.

## Notes
- All evaluations are performed at **640×640** input resolution.
- Model weights are loaded from the `models/` folder.
