# dataset.py
import os, glob, random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def load_nii(path, dtype=np.float32):
    img = nib.load(path).get_fdata(caching='unchanged')
    if img.ndim == 4:  # (X,Y,Z,1) -> (X,Y,Z)
        img = img[..., 0]
    return img.astype(dtype)

def default_name_mapper(img_name: str, img_token='LFOV', lab_token='SEMANTIC'):
    """R039_Week2_LFOV.nii.gz -> R039_Week2_SEMANTIC.nii.gz"""
    return (img_name
            .replace(f"_{img_token}.nii.gz", f"_{lab_token}.nii.gz")
            .replace(f"_{img_token}.nii",    f"_{lab_token}.nii"))

class HipMRI3DNIIDataset(Dataset):
    """
    root/
      <image_dir>/*.nii.gz      (e.g. semantic_MRs, 带 *_LFOV*)
      <label_dir>/*.nii.gz      (e.g. semantic_labels_only, 带 *_SEMANTIC*)

    重要参数：
      - binary_prostate: True 时把标签二值化 (prostate=1, others=0)
      - prostate_label : 前列腺在原始标签中的类别值（默认 5）
    """
    def __init__(self, root, image_dir='images', label_dir='labels',
                 norm='zscore', patch=None, augment=False, classes=5,
                 img_token='LFOV', lab_token='SEMANTIC',
                 binary_prostate=False, prostate_label=5):
        self.root = root
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.norm = norm
        self.patch = patch
        self.augment = augment
        self.classes = classes
        self.img_token = img_token
        self.lab_token = lab_token
        self.binary_prostate = binary_prostate
        self.prostate_label = int(prostate_label)

        img_glob = os.path.join(root, image_dir, '*.nii*')
        all_imgs = sorted(glob.glob(img_glob))
        pairs = []
        for ip in all_imgs:
            base = os.path.basename(ip)
            lab_base = default_name_mapper(base, img_token, lab_token)
            lp = os.path.join(root, label_dir, lab_base)
            if os.path.exists(lp):
                pairs.append((ip, lp))
            else:
                alt = os.path.join(root, label_dir, base)
                if os.path.exists(alt):
                    pairs.append((ip, alt))

        self.pairs = pairs
        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f'No paired volumes found.\nimages: {img_glob}\nlabels dir: {os.path.join(root,label_dir)}\n'
                f'token map: {img_token}->{lab_token}'
            )

        print(f'Found {len(self.pairs)} paired volumes. Example pairs:')
        for i in range(min(3, len(self.pairs))):
            print('  IMG:', self.pairs[i][0])
            print('  LAB:', self.pairs[i][1])

    def __len__(self): return len(self.pairs)

    def _norm(self, x):
        if self.norm == 'zscore':
            m, s = x.mean(), x.std() + 1e-6
            return (x - m) / s
        elif self.norm == 'minmax':
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
        return x

    def _random_crop3d(self, img, lab, size):
        sd, sh, sw = size
        d, h, w = img.shape
        pd, ph, pw = max(0, sd-d), max(0, sh-h), max(0, sw-w)
        if pd or ph or pw:
            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            lab_t = torch.from_numpy(lab).unsqueeze(0).unsqueeze(0)
            img = F.pad(img_t, [pw//2, pw-pw//2, ph//2, ph-ph//2, pd//2, pd-pd//2]).squeeze().numpy()
            lab = F.pad(lab_t, [pw//2, pw-pw//2, ph//2, ph-ph//2, pd//2, pd-pd//2]).squeeze().numpy()
            d, h, w = img.shape
        z = np.random.randint(0, d - sd + 1)
        y = np.random.randint(0, h - sh + 1)
        x = np.random.randint(0, w - sw + 1)
        return img[z:z+sd, y:y+sh, x:x+sw], lab[z:z+sd, y:y+sh, x:x+sw]

    def __getitem__(self, idx):
        ipath, lpath = self.pairs[idx]
        img = load_nii(ipath)               # (X,Y,Z)
        lab = load_nii(lpath, np.uint8)     # (X,Y,Z)

        # -> (D,H,W) = (Z,Y,X)
        img = np.transpose(img, (2,0,1))
        lab = np.transpose(lab, (2,0,1))

        img = self._norm(img)

        if self.binary_prostate:
            # 前列腺=1，其它=0
            lab = (lab == self.prostate_label).astype(np.int64)

        if self.patch is not None:
            img, lab = self._random_crop3d(img, lab, self.patch)

        img = torch.from_numpy(img).unsqueeze(0).float()  # (1,D,H,W)
        lab = torch.from_numpy(lab).long()                # (D,H,W)

        if self.augment:
            if random.random() < 0.5: img = torch.flip(img, [1]); lab = torch.flip(lab, [0])
            if random.random() < 0.5: img = torch.flip(img, [2]); lab = torch.flip(lab, [1])
            if random.random() < 0.5: img = torch.flip(img, [3]); lab = torch.flip(lab, [2])

        return img, lab, os.path.basename(ipath)
