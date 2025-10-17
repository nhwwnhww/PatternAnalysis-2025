import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    多分类/二分类通用 Dice loss。
    logits: (N, C, H, W)
    target: (N,H,W) 或 (N,1,H,W) 的整型标签
    """
    assert logits.dim() == 4
    N, C, H, W = logits.shape

    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    target = target.long()

    if C == 1:
        probs = torch.sigmoid(logits)              # (N,1,H,W)
        tgt = target.float().unsqueeze(1)          # (N,1,H,W)
        inter = (probs * tgt).sum((0,2,3))
        union = probs.sum((0,2,3)) + tgt.sum((0,2,3))
        dice = (2*inter + eps) / (union + eps)
        return 1 - dice.mean()

    probs = F.softmax(logits, dim=1)               # (N,C,H,W)
    tgt_1h = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()  # (N,C,H,W)
    inter = (probs * tgt_1h).sum((0,2,3))
    union = probs.sum((0,2,3)) + tgt_1h.sum((0,2,3))
    dice  = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, exclude_bg: bool = True, eps: float = 1e-6) -> float:
    """
    评估用 Dice（对 batch 求平均）。多分类使用 argmax。
    """
    N, C, H, W = logits.shape
    if targets.dim() == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    targets = targets.long()

    if C == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        tgt = targets.float().unsqueeze(1)
        inter = (preds * tgt).sum((1,2,3))
        denom = preds.sum((1,2,3)) + tgt.sum((1,2,3))
        dice = ((2*inter + eps) / (denom + eps)).mean()
        return dice.item()

    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)  # (N,H,W)
    preds_oh   = F.one_hot(preds, num_classes=C).permute(0,3,1,2).float()
    targets_oh = F.one_hot(targets, num_classes=C).permute(0,3,1,2).float()
    inter = (preds_oh * targets_oh).sum((0,2,3))
    denom = preds_oh.sum((0,2,3)) + targets_oh.sum((0,2,3))
    dice_c = (2*inter + eps) / (denom + eps)
    if exclude_bg and C > 1:
        dice_c = dice_c[1:]
    return dice_c.mean().item()

def double_conv(ic, oc, act_layer=nn.SiLU):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc), act_layer(),
        nn.Conv2d(oc, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc), act_layer(),
    )

class UpBlock(nn.Module):
    """上采样 -> 与对应跳连拼接 -> 双卷积；确保空间尺寸与 skip 对齐"""
    def __init__(self, in_ch, skip_ch, out_ch, act_layer=nn.SiLU):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = double_conv(in_ch + skip_ch, out_ch, act_layer)

    def forward(self, x, skip):
        x = self.up(x)
        # pad 以对齐（偶数/奇数尺寸时很有用）
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    """
    对称的 3-down / 3-up 结构（输入/输出同尺寸）:
      enc:  b, 2b, 4b, 8b
      bot:  16b
      dec:  (16b ->+4b -> 8b) -> (8b ->+2b -> 4b) -> (4b ->+b -> 2b)
    """
    def __init__(self, in_c=1, n_classes=4, base=32, act_layer=nn.SiLU):
        super().__init__()
        # encoder
        self.inc   = double_conv(in_c, base, act_layer)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(base,   base*2, act_layer))  # /2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(base*2, base*4, act_layer))  # /4
        self.down3 = nn.Sequential(nn.MaxPool2d(2), double_conv(base*4, base*8, act_layer))  # /8
        # bottleneck（不再继续下采样）
        self.bot   = double_conv(base*8, base*16, act_layer)  # /8
        # decoder（注意每一级与对应的 encoder 层对齐）
        self.up1 = UpBlock(in_ch=base*16, skip_ch=base*4, out_ch=base*8,  act_layer=act_layer)  # -> /4  拼 x3
        self.up2 = UpBlock(in_ch=base*8,  skip_ch=base*2, out_ch=base*4,  act_layer=act_layer)  # -> /2  拼 x2
        self.up3 = UpBlock(in_ch=base*4,  skip_ch=base,   out_ch=base*2,  act_layer=act_layer)  # -> /1  拼 x1
        self.outc = nn.Conv2d(base*2, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)        # /1   (b)
        x2 = self.down1(x1)     # /2   (2b)
        x3 = self.down2(x2)     # /4   (4b)
        x4 = self.down3(x3)     # /8   (8b)
        xb = self.bot(x4)       # /8   (16b)

        x = self.up1(xb, x3)    # /4   → 8b
        x = self.up2(x,  x2)    # /2   → 4b
        x = self.up3(x,  x1)    # /1   → 2b
        return self.outc(x)     # /1   (原始分辨率)
