# dataset.py
import os, glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def _load_nii(path):
    img = nib.load(path).get_fdata(caching='unchanged')
    if img.ndim == 3:  # (H,W,1)
        img = img[:,:,0]
    return img.astype(np.float32)

class HipMRI2DNIIDataset(Dataset):
    """
    期望目录结构（Rangpur/本地）：
    root/
      images/ *.nii or *.nii.gz
      labels/ *.nii or *.nii.gz
    文件名需一一对应（同名）。
    """
    def __init__(self, root, image_dir='images', label_dir='labels',
                 norm='zscore', augment=False, classes=5):
        self.ipaths = sorted(glob.glob(os.path.join(root, image_dir, '*.nii*')))
        self.lpaths = [os.path.join(root, label_dir, os.path.basename(p)) for p in self.ipaths]
        self.norm   = norm
        self.aug    = augment
        self.classes = classes
        assert len(self.ipaths) == len(self.lpaths) and len(self.ipaths) > 0, "Empty or mismatched dataset."

    def __len__(self): return len(self.ipaths)

    def __getitem__(self, idx):
        ipath, lpath = self.ipaths[idx], self.lpaths[idx]
        img  = _load_nii(ipath)   # (H,W)
        mask = _load_nii(lpath)   # (H,W), integer labels [0..C-1]

        # normalize
        if self.norm == 'zscore':
            m, s = img.mean(), img.std() + 1e-6
            img = (img - m) / s
        elif self.norm == 'minmax':
            mn, mx = np.percentile(img, 1), np.percentile(img, 99)
            img = np.clip((img - mn) / (mx - mn + 1e-6), 0, 1)

        img  = torch.from_numpy(img).unsqueeze(0)     # (1,H,W)
        mask = torch.from_numpy(mask).long()          # (H,W)

        # simple aug
        if self.aug:
            if torch.rand(1).item() < 0.5:
                img  = TF.hflip(img);  mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
            if torch.rand(1).item() < 0.5:
                img  = TF.vflip(img);  mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        return img, mask, os.path.basename(ipath)
