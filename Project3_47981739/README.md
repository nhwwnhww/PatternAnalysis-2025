# HipMRI 2D Segmentation (Improved UNet, s4798173)

## Task
Segment HipMRI 2D slices (prostate study) with an Improved 2D UNet. Target: prostate label Dice ≥ 0.75 on test set.  
Project spec & required files: see COMP3710 handout.  
- Required files and plotting during training: 【spec §4】.
- Dataset & NIfTI note for Project 3: 【spec §1.3/3】.

## Data
Rangpur: `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`  
Prepare two folders:
`data_root/
images/.nii.gz
labels/.nii.gz`

## Train
```
python train.py --data_root /path/to/data_root 
                --out_dir ./runs/hipmri2d_unet --epochs 40 --batch 8
```

## Predict & Visualize

```
python predict.py --data_root /path/to/data_root 
                  --weights runs/hipmri2d_unet/best.pt 
                  --out_dir ./pred_vis --num 12
```

Files

- modules.py : Improved UNet2D + Dice

- dataset.py : HipMRI 2D NIfTI loader

- train.py : train/val/test & curves

- predict.py : inference + overlays

- README.md : instructions