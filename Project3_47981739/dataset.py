# dataset.py
import os, glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def _load_nii(path):
    """Load a NIfTI file as a 2D float32 numpy array."""
    img = nib.load(path).get_fdata(caching='unchanged')
    if img.ndim == 3:  # (H, W, 1)
        img = img[:, :, 0]
    return img.astype(np.float32)


class HipMRI2DNIIDataset(Dataset):
    """
    Expects directory structure:
    root/
        images/ *.nii or *.nii.gz
        labels/ *.nii or *.nii.gz
    Filenames must match between images and labels.
    """

    def __init__(self, root, image_dir='images', label_dir='labels',
                 norm='zscore', augment=False, classes=5):
        # auto-detect keras_slices_train / keras_slices_seg_train
        if os.path.exists(os.path.join(root, 'keras_slices_train')):
            image_dir = 'keras_slices_train'
        if os.path.exists(os.path.join(root, 'keras_slices_seg_train')):
            label_dir = 'keras_slices_seg_train'

        self.ipaths = sorted(glob.glob(os.path.join(root, image_dir, '*.nii*')))
        self.lpaths = sorted(glob.glob(os.path.join(root, label_dir, '*.nii*')))

        assert len(self.ipaths) == len(self.lpaths) and len(self.ipaths) > 0, \
            f"Empty or mismatched dataset: {len(self.ipaths)} images, {len(self.lpaths)} labels"

        self.norm = norm
        self.aug = augment
        self.classes = classes

        # automatically detect max H and W for padding
        max_h, max_w = 0, 0
        for ipath in self.ipaths:
            img = _load_nii(ipath)
            h, w = img.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        self.target_size = (max_h, max_w)

    def __len__(self):
        return len(self.ipaths)

    def __getitem__(self, idx):
        ipath, lpath = self.ipaths[idx], self.lpaths[idx]
        img = _load_nii(ipath)   # (H, W)
        mask = _load_nii(lpath)  # (H, W), integer labels [0..C-1]

        # normalize
        if self.norm == 'zscore':
            m, s = img.mean(), img.std() + 1e-6
            img = (img - m) / s
        elif self.norm == 'minmax':
            mn, mx = np.percentile(img, 1), np.percentile(img, 99)
            img = np.clip((img - mn) / (mx - mn + 1e-6), 0, 1)

        # convert to torch
        img = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)
        mask = torch.from_numpy(mask).long()      # (H,W)

        target_h, target_w = self.target_size

        # resize image via interpolate
        img = F.interpolate(img.unsqueeze(0), size=(target_h, target_w),
                            mode='bilinear', align_corners=False).squeeze(0)

        # pad mask to target size (safe for integer labels)
        pad_h = target_h - mask.shape[0]
        pad_w = target_w - mask.shape[1]
        mask = F.pad(mask, (0, pad_w, 0, pad_h), value=0)

        # simple augmentation
        if self.aug:
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
            if torch.rand(1).item() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        return img, mask, os.path.basename(ipath)

