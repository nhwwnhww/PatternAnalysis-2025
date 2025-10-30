# 3D Prostate MRI Segmentation using Improved UNet
> COMP3710 -- Pattern Analysis 
> Student: s47981739

### Problem Description
This project addresses automatic 3D prostate segmentation on pelvic MRI using a
T2-weighted large-FOV (LFOV) volume as input, and producing a binary mask.
**Goal**: produce a binary mask of prostate (foreground=1, background=0), then evaluate with Dice.

Dataset: **HipMRI_Study_open** (ours), using `semantic_MRs/*_LFOV.nii.gz` as `images and semantic_labels_only/*_SEMANTIC.nii.gz` as labels.
We treat it as **binary** by remapping the original “prostate class=5” → 1 and others → 0.

### Method & Architecture
We adopt a **3D Improved U-Net** with encoder–decoder skip connections and deep
supervision (inherited from provided course template & self-implemented modules).
Training is supervised with Dice-based loss to maximize volume overlap with labels.

For inference, we apply:

- Sliding-window prediction to avoid GPU OOM on full 3D

- Test-time augment (TTA) with flip ensemble

- Largest Connected Component (LCC) post-processing to suppress noise

- Week-adaptive thresholding for softmax probability to refine foreground mask

### Data & Pre-processing
**Input**: `semantic_MRs/*_LFOV.nii.gz`

**GT**: `semantic_labels_only/*_SEMANTIC.nii.gz`

**Normalization**: clip to [0.5, 99.5] percentile (within non-zero region), z-score

**Cropping**: No cropping required (LFOV used directly)

Output space equals original affine (prediction saved in native geometry)

### Training Setup

**Model**: 3D U-Net (Improved)

**Loss**: DiceLoss (binary mode)

**Optimizer**: Adam

**Scheduler**: ReduceLROnPlateau

**Epochs**: converged ~ last 10 epochs (see loss curve)

**Final checkpoint used**: `runs/hipmri3d_unet_bin/best.pt`
![dice](.\runs\hipmri3d_unet_bin\dice.png)
![lose](.\runs\hipmri3d_unet_bin\loss.png)

### Inference & Reproducibility (with commands)
Example — single case with SW+TTA+LCC:
```
python predict.py ^
  --image "C:\COMP3710\HipMRI_Study_open\semantic_MRs\B006_Week0_LFOV.nii.gz" ^
  --weights ".\runs\hipmri3d_unet_bin\best.pt" ^
  --out_dir ".\Predict_Result\B006_Week0" ^
  --label_root "C:\COMP3710\HipMRI_Study_open" ^
  --label_dir "semantic_labels_only" ^
  --binary_prostate --postprocess_lcc --save_nii --prob --metrics_csv ^
  --softmax_thr 0.20 --prob_class 1 ^
  --sw_enable --sw_patch "96,128,128" --sw_stride "48,64,64" ^
  --tta --grid --slices "42,64,85"
```
A `.bat` script was additionally prepared to batch all weeks with **week-based thresholds**
(W0–1:0.25, W2–4:0.20, W5–7:0.18).

### Results (single case + global stats)

#### Global Dice (422 cases, no SW+TTA):
Mean Dice = 0.577 ± 0.229
![dice_box](.\result_summary\dice_box.png)
![dice_by_week](.\result_summary\dice_by_week.png)
![dice_hist](.\result_summary\dice_hist.png)

### Analysis and Observations

Although the model reaches ~0.58 Dice on average without SW+TTA, it does not meet
the 0.70+ target yet. The under-performance is explainable:

- Strong anatomical variability across weeks (post-treatment changes)

- Distribution shift between training weeks and evaluation weeks

- Thresholding sensitivity — Dice dropped significantly when threshold changes

- Noise leaks at apex/base even after LCC, typical failure region in prostate MRI

- Sliding-window/TTA effects were not yet reflected in the baseline summary

Therefore the new SW+TTA+week-thr pipeline is re-running and expected to push
mean Dice closer to 0.70 when aggregated.

### Acknowledgements
Base code and training loop authored by the student per assignment, leveraging PyTorch. Model design and losses are implemented in this repository. 