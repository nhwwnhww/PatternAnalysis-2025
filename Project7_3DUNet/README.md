# Prostate 3D Segmentation with Improved 3D UNet (s4798173)

**Task (Project 7)**: Segment the *downsampled Prostate 3D dataset* with a 3D (Improved) UNet.  
Target: **Dice ≥ 0.7** on test set. You may begin with the original 3D UNet (Çiçek et al., MICCAI 2016) and load **NIfTI** as per Appendix B.  
Spec refs: Project description & difficulty notes; required files list.  
- Project 7 description & Dice target: 【handout §1.3/7】  
- NIfTI reading examples (Appendix B): 【handout Appendix B】  
- Required files/plotting: 【handout §4】.


## Data (Rangpur)
- Images: `/home/groups/comp3710/HipMRI_Study_open/semantic_MRs`
- Labels: `/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only`

## Data (Local)
- Images: `D:\COMP3710\HipMRI_Study_open/semantic_MRs`
- Labels: `D:\COMP3710\HipMRI_Study_open/semantic_labels_only`

## Train

### bin
```
python train.py 
  --data_root "D:\COMP3710\HipMRI_Study_open" 
  --image_dir "semantic_MRs" 
  --label_dir "semantic_labels_only" 
  --out_dir ".\runs\hipmri3d_unet_bin" 
  --epochs 10 
  --batch 1 
  --patch 96,160,160 
  --binary_prostate 
  --prostate_label 5 
  --classwise
```

## Predict (single volume)

### bin
```
python predict.py
  --image "D:\COMP3710\HipMRI_Study_open\semantic_MRs\B006_Week0_LFOV.nii.gz"
  --weights ".\runs\hipmri3d_unet_bin\best.pt"
  --out_dir ".\pred3d_vis_bin"
  --label_root "D:\COMP3710\HipMRI_Study_open"
  --label_dir "semantic_labels_only"
  --binary_prostate
  --metrics_csv --grid --tta --postprocess_lcc --save_nii
  --prob 
```
