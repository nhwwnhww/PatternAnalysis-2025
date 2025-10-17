# predict.py (compatible + rich visualization)
import os, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import amp as torch_amp
from torch.utils.data import DataLoader
from dataset import OASIS2DPNGDataset
from modules import UNet2D, dice_score

def build_loader(data_root, split, img_size, num_classes, batch_size, num_workers):
    if split == "test":
        img_dir = os.path.join(data_root, "keras_png_slices_test")
        msk_dir = os.path.join(data_root, "keras_png_slices_seg_test")
    elif split in ("val", "validate"):
        img_dir = os.path.join(data_root, "keras_png_slices_validate")
        msk_dir = os.path.join(data_root, "keras_png_slices_seg_validate")
    elif split == "train":
        img_dir = os.path.join(data_root, "keras_png_slices_train")
        msk_dir = os.path.join(data_root, "keras_png_slices_seg_train")
    else:
        raise ValueError(f"Unknown split: {split}")

    ds = OASIS2DPNGDataset(img_dir, msk_dir, img_size=img_size, augment=False, num_classes=num_classes)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    return ds, dl

def load_weights(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)  # 优先安全模式
    except TypeError:
        print("[warn] torch.load(weights_only=True) not supported; falling back to weights_only=False")
        ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model", ckpt)  # 兼容 {"model": sd} 或直接 sd
    model.load_state_dict(state_dict)
    return model

def _colorize_mask(mask, num_classes):
    """将整数标签(0..C-1)映射成RGB，便于overlay。"""
    import matplotlib
    cmap = matplotlib.cm.get_cmap('tab20', num_classes)
    rgb = cmap(mask / max(1, num_classes-1))[..., :3]  # (H,W,3)
    rgb[mask == 0] = np.array([0.2, 0.2, 0.2])        # 背景更暗
    return rgb

@torch.no_grad()
def visualize_batch(model, loader, num_classes, device, n=3, save_dir=None,
                    overlay=True, alpha=0.5, contour=False, indices=None, show_legend=True):
    """
    可视化 n 个样本：Image / Mask(GT) / Prediction
    - overlay:   灰度图+半透明伪彩mask
    - contour:   画等高线轮廓（更干净）
    - indices:   指定样本索引（相对于当前 batch）
    - save_dir:  若非空则保存PNG到该目录
    """
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.to(device); masks = masks.to(device)
    with torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
        logits = model(imgs); preds = logits.argmax(1)

    imgs  = imgs.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    N = imgs.shape[0]
    pick = indices if (indices and len(indices) > 0) else list(range(min(n, N)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in pick:
        img = imgs[i, 0]
        msk = masks[i]
        prd = preds[i]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        # Image
        axes[0].imshow(img, cmap='gray'); axes[0].set_title("Image"); axes[0].axis('off')
        # GT
        if overlay:
            axes[1].imshow(img, cmap='gray')
            axes[1].imshow(_colorize_mask(msk, num_classes), alpha=alpha)
        else:
            axes[1].imshow(msk, cmap='jet', vmin=0, vmax=num_classes-1)
        if contour:
            axes[1].contour(msk, levels=np.unique(msk)[1:], linewidths=1)
        axes[1].set_title("Mask (GT)"); axes[1].axis('off')
        # Pred
        if overlay:
            axes[2].imshow(img, cmap='gray')
            axes[2].imshow(_colorize_mask(prd, num_classes), alpha=alpha)
        else:
            axes[2].imshow(prd, cmap='jet', vmin=0, vmax=num_classes-1)
        if contour:
            axes[2].contour(prd, levels=np.unique(prd)[1:], linewidths=1)
        axes[2].set_title("Prediction"); axes[2].axis('off')

        if show_legend:
            import matplotlib.patches as mpatches
            handles = []
            for c in range(num_classes):
                color = _colorize_mask(np.full((1,1), c, dtype=np.int32), num_classes)[0,0]
                handles.append(mpatches.Patch(color=color, label=f"class{c}"))
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(num_classes, 6), bbox_to_anchor=(0.5,-0.02))

        plt.tight_layout()
        if save_dir:
            out = os.path.join(save_dir, f"viz_{i:03d}.png")
            plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()

@torch.no_grad()
def evaluate_per_class_dice(model, loader, num_classes: int, device: torch.device):
    """基于混淆矩阵计算 per-class Dice：2TP/(2TP+FP+FN)，显存友好"""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    model.eval()
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
        k = (masks * num_classes + preds).view(-1)
        cm += torch.bincount(k, minlength=num_classes*num_classes).view(num_classes, num_classes)
    TP = cm.diag().float()
    FP = cm.sum(0).float() - TP
    FN = cm.sum(1).float() - TP
    dice_c = (2*TP) / (2*TP + FP + FN + 1e-6)
    return dice_c.cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--weights",   type=str, required=True)
    ap.add_argument("--img_size",  type=int, default=128)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--split", type=str, default="test", choices=["test","validate","val","train"])
    ap.add_argument("--no_viz", action="store_true", help="禁用样本可视化")
    # ↓↓↓ 新增可视化参数 ↓↓↓
    ap.add_argument("--viz_n", type=int, default=3, help="可视化样本数量")
    ap.add_argument("--viz_idx", type=int, nargs="*", default=None, help="指定样本索引（相对于一个batch）")
    ap.add_argument("--viz_overlay", action="store_true", help="叠加半透明mask在灰度图上")
    ap.add_argument("--viz_contour", action="store_true", help="绘制mask等高线轮廓")
    ap.add_argument("--viz_alpha", type=float, default=0.5, help="overlay透明度")
    ap.add_argument("--save_dir", type=str, default=None, help="保存可视化图片的文件夹")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    ds, loader = build_loader(args.data_root, args.split, args.img_size,
                              args.num_classes, args.batch_size, args.num_workers)

    # 模型
    model = UNet2D(in_c=1, n_classes=args.num_classes, base=32).to(device)
    model = load_weights(model, args.weights, device)

    # 总体 Dice（排除背景）——与训练阶段 dice_score 对齐
    model.eval()
    total, ssum = 0, 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch_amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
                logits = model(imgs)
            ssum += dice_score(logits, masks) * imgs.size(0)
            total += imgs.size(0)
    print(f"[{args.split}] Dice (exclude bg): {ssum/total:.4f}")

    # 每类 Dice + 达标判定（Project1: 所有前景类 ≥ 0.90）
    dice_c = evaluate_per_class_dice(model, loader, args.num_classes, device)
    print("Per-class Dice (incl bg):", [f"{v:.4f}" for v in dice_c])
    if args.num_classes > 1:
        fg = dice_c[1:]
        print("Per-class Dice (excl bg):", [f"{v:.4f}" for v in fg])
        passed = bool((fg >= 0.90).all().item())
        print("PASS (all foreground classes ≥ 0.90):", passed, "| min_fg =", float(fg.min()))

    # 可视化
    if not args.no_viz:
        visualize_batch(model, loader, args.num_classes, device,
                        n=args.viz_n,
                        save_dir=args.save_dir,
                        overlay=args.viz_overlay,
                        alpha=args.viz_alpha,
                        contour=args.viz_contour,
                        indices=args.viz_idx,
                        show_legend=True)

if __name__ == "__main__":
    main()
