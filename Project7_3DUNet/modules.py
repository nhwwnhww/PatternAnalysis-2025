# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Metrics & Loss
# --------------------
def dice_score_3d(probs, target, eps=1e-6):
    """
    probs : (N,C,D,H,W) softmax
    target: (N,D,H,W) int64  or one-hot (N,C,D,H,W)
    returns macro Dice over classes
    """
    if target.ndim == 4:
        onehot = F.one_hot(target.long(), num_classes=probs.size(1)).permute(0,4,1,2,3).float()
    else:
        onehot = target.float()
    num = 2 * (probs * onehot).sum(dim=(0,2,3,4))
    den = (probs + onehot).sum(dim=(0,2,3,4)).clamp_min(eps)
    return (num / den).mean()

class DiceLoss3D(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        C = logits.size(1)
        # —— 关键：先检查标签是否越界（避免 CUDA 断言）
        max_label = int(target.max().item())
        if max_label >= C:
            raise RuntimeError(
                f"[DiceLoss3D] label value {max_label} >= num_classes {C}. "
                f"Try launching with --classes {max_label+1} (e.g., 6 for BG+5 organs), "
                f"or remap labels."
            )
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), num_classes=C).permute(0,4,1,2,3).float()
        num = 2 * (probs * onehot).sum(dim=(0,2,3,4))
        den = (probs + onehot).sum(dim=(0,2,3,4)).clamp_min(self.smooth)
        dice = (num + self.smooth) / (den + self.smooth)
        return 1 - dice.mean()


# --------------------
# Improved UNet 3D
#   - Residual conv block
#   - Squeeze-and-Excite (channel attention)
# --------------------
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.drop  = nn.Dropout3d(drop) if drop>0 else nn.Identity()
        self.res   = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch!=out_ch else nn.Identity()
        # SE
        self.se1   = nn.Conv3d(out_ch, max(1,out_ch//8), 1)
        self.se2   = nn.Conv3d(max(1,out_ch//8), out_ch, 1)

    def forward(self, x):
        identity = self.res(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.drop(F.relu(self.bn2(self.conv2(x)), inplace=True))
        w = F.adaptive_avg_pool3d(x, 1)
        w = F.relu(self.se1(w), inplace=True)
        w = torch.sigmoid(self.se2(w))
        x = x * w
        return x + identity

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ConvBlock3D(in_ch, out_ch, drop)
    def forward(self, x): return self.block(self.pool(x))

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = ConvBlock3D(in_ch, out_ch, drop)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        ds = [skip.size(i) - x.size(i) for i in (2,3,4)]
        if any(d!=0 for d in ds):
            x = F.pad(x, [ds[2]//2, ds[2]-ds[2]//2, ds[1]//2, ds[1]-ds[1]//2, ds[0]//2, ds[0]-ds[0]//2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class UNet3D_Improved(nn.Module):
    def __init__(self, in_ch=1, n_classes=5, base=16, drop=0.1):
        super().__init__()
        self.inc = ConvBlock3D(in_ch, base, drop)
        self.d1  = Down3D(base, base*2, drop)
        self.d2  = Down3D(base*2, base*4, drop)
        self.d3  = Down3D(base*4, base*8, drop)
        self.bot = ConvBlock3D(base*8, base*16, drop)
        self.u3  = Up3D(base*16, base*8, drop)
        self.u2  = Up3D(base*8,  base*4, drop)
        self.u1  = Up3D(base*4,  base*2, drop)
        self.u0  = Up3D(base*2,  base,   drop)
        self.out = nn.Conv3d(base, n_classes, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        xb = self.bot(x3)
        x  = self.u3(xb, x3)
        x  = self.u2(x,  x2)
        x  = self.u1(x,  x1)
        x  = self.u0(x,  x0)
        return self.out(x)
