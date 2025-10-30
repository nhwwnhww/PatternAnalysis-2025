import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Metrics / Losses
# -------------------------
def dice_score(pred, target, eps=1e-6):
    """
    pred: (N,C,H,W) after softmax
    target: (N,H,W)  int64  or one-hot (N,C,H,W)
    """
    if target.ndim == 3:
        target_1h = F.one_hot(target.long(), num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
    else:
        target_1h = target.float()
    num = 2 * (pred * target_1h).sum(dim=(0,2,3))
    den = (pred + target_1h).sum(dim=(0,2,3)).clamp_min(eps)
    dice_c = (num / den)
    return dice_c.mean()  # macro over classes

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        if target.ndim == 3:
            target_1h = F.one_hot(target.long(), num_classes=logits.size(1)).permute(0,3,1,2).float()
        else:
            target_1h = target.float()
        num = 2 * (probs * target_1h).sum(dim=(0,2,3))
        den = (probs + target_1h).sum(dim=(0,2,3)).clamp_min(self.smooth)
        dice = (num + self.smooth) / (den + self.smooth)
        return 1 - dice.mean()

# -------------------------
# Improved UNet 2D (Residual + Squeeze&Excite)
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.res   = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        # Squeeze & Excite
        self.se_fc1 = nn.Conv2d(out_ch, out_ch//8, 1)
        self.se_fc2 = nn.Conv2d(out_ch//8, out_ch, 1)

    def forward(self, x):
        identity = self.res(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.drop(F.relu(self.bn2(self.conv2(x)), inplace=True))
        # SE
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.se_fc1(w), inplace=True)
        w = torch.sigmoid(self.se_fc2(w))
        x = x * w
        return x + identity

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch, drop)
    def forward(self, x):
        return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = ConvBlock(in_ch, out_ch, drop)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class UNet2D_Improved(nn.Module):
    def __init__(self, in_ch=1, n_classes=5, base=32, drop=0.1):
        super().__init__()
        self.inc = ConvBlock(in_ch, base, drop=drop)
        self.d1  = Down(base, base*2, drop=drop)
        self.d2  = Down(base*2, base*4, drop=drop)
        self.d3  = Down(base*4, base*8, drop=drop)
        self.bot = ConvBlock(base*8, base*16, drop=drop)
        self.u3  = Up(base*16, base*8, drop=drop)
        self.u2  = Up(base*8,  base*4, drop=drop)
        self.u1  = Up(base*4,  base*2, drop=drop)
        self.u0  = Up(base*2,  base,   drop=drop)
        self.out = nn.Conv2d(base, n_classes, 1)

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
