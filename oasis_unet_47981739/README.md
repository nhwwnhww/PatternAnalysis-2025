# OASIS 2D Brain Segmentation — UNet (Project 1)

**Goal:** Segment the OASIS 2D slices; achieve **Dice ≥ 0.90** on all labels (test set).  
**Model:** UNet2D (base=32), loss = Dice loss, AMP training.

## Environment
- Python 3.10+
- PyTorch 2.5+, CUDA (optional)
- matplotlib, tqdm, pillow, torchvision

## Data
Path structure:

## train code
```
python train.py `  --data_root D:/COMP3710/OASIS `  --save_dir C:\Users\nhwen\OneDrive\Desktop\PatternAnalysis-2025\oasis_unet_47981739\runs_oasis_unet `  --epochs 20 --img_size 128 --num_classes 4 --batch_size 8
```

## validate and visualize

### print validate value only
```
python predict.py --data_root D:/COMP3710/OASIS `
  --weights "C:\Users\nhwen\OneDrive\Desktop\PatternAnalysis-2025\oasis_unet_47981739\runs_oasis_unet\best.pt" `
  --split validate --no_viz --img_size 128 --num_classes 4 --batch_size 8
```
### validate value and visulize

### save 6 png to viz_test (Overlay + Outline)
```
python predict.py `
  --data_root D:/COMP3710/OASIS `
  --weights "C:\Users\nhwen\OneDrive\Desktop\PatternAnalysis-2025\oasis_unet_47981739\runs_oasis_unet\best.pt" `
  --split test --viz_overlay --viz_contour --viz_alpha 0.6 `
  --viz_n 6 --save_dir ".\viz_test"

```

### look for idx 0,5,12
```
python predict.py --data_root D:/COMP3710/OASIS --weights "C:\Users\nhwen\OneDrive\Desktop\PatternAnalysis-2025\oasis_unet_47981739\runs_oasis_unet\best.pt" `
  --split validate --viz_idx 0 5 12 --viz_overlay --viz_alpha 0.5
```

###  Traditional 3-in-a-row (non-stacking)
```
python predict.py --data_root D:/COMP3710/OASIS --weights "C:\Users\nhwen\OneDrive\Desktop\PatternAnalysis-2025\oasis_unet_47981739\runs_oasis_unet\best.pt" --viz_n 3
```