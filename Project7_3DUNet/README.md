# 3D Prostate MRI Segmentation
> COMP3710 Pattern Analysis — Topic Recognition
> Student: s47981739 (replace with your name/ID if needed)

### Problem
Automatic prostate segmentation on 3D T2-weighted MR volumes.
**Goal**: produce a binary mask of prostate (foreground=1, background=0), then evaluate with Dice.

Dataset: **HipMRI_Study_open** (ours), using `semantic_MRs/*_LFOV.nii.gz` as `images and semantic_labels_only/*_SEMANTIC.nii.gz` as labels.
We treat it as **binary** by remapping the original “prostate class=5” → 1 and others → 0.

### Method

### How to run

### Results

### Analysis

### Repro checklist (aligns with course brief)

### Acknowledgements
Base code and training loop authored by the student per assignment, leveraging PyTorch. Model design and losses are implemented in this repository. 