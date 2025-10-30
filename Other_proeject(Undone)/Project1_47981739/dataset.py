# dataset.py
import os, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class OASIS2DPNGDataset(Dataset):
    """
    OASIS PNG 2D 切片分割数据集
    - image_dir: 灰度 PNG
    - mask_dir : PNG（像素值需能映射到 0..NUM_CLASSES-1）
    - augment  : 随机翻转
    - img_size : 统一 resize 到正方形
    """
    def __init__(self, image_dir, mask_dir, img_size=128, augment=False, num_classes=4):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.img_files = sorted(os.listdir(image_dir))
        self.mask_files= sorted(os.listdir(mask_dir))
        assert len(self.img_files) == len(self.mask_files), "图像/掩码数量不一致"
        self.img_size  = img_size
        self.augment   = augment
        self.num_classes = num_classes

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.img_files[idx])).convert("L")
        mask= Image.open(os.path.join(self.mask_dir,  self.mask_files[idx])).convert("L")

        img  = TF.resize(img,  [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        img  = TF.to_tensor(img)  # (1,H,W), float32 [0,1]
        mask = np.array(mask, dtype=np.uint8)

        # 你之前的做法：把灰度值压成 {0..C-1}
        # 如果你的 mask 是 0/85/170/255 这种，下面这句就行：
        mask = (mask // (256 // self.num_classes)).astype(np.int64)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img); mask = np.ascontiguousarray(np.fliplr(mask))
            if random.random() < 0.5:
                img = TF.vflip(img); mask = np.ascontiguousarray(np.flipud(mask))

        mask = torch.from_numpy(mask).long()
        return img, mask
