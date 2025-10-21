# predict_debug.py — 3D UNet 验收/排错增强版（稳定版）
import os, argparse, json, csv
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from modules import UNet3D_Improved


# ------------------ 基础工具 ------------------
def to_dhw(vol):
    """(X,Y,Z[,(1)]) -> (D,H,W)=(Z,Y,X)"""
    if vol.ndim == 4:
        vol = vol[..., 0]
    return np.transpose(vol, (2, 0, 1))

def from_dhw(vol_dhw):
    """(D,H,W) -> (X,Y,Z)"""
    return np.transpose(vol_dhw, (2, 1, 0))

def default_name_mapper(img_name: str, img_token='LFOV', lab_token='SEMANTIC'):
    """R039_Week2_LFOV.nii.gz -> R039_Week2_SEMANTIC.nii.gz"""
    return (img_name
            .replace(f"_{img_token}.nii.gz", f"_{lab_token}.nii.gz")
            .replace(f"_{img_token}.nii",    f"_{lab_token}.nii"))

def dice_per_class_np(pred, gt, num_classes, eps=1e-6):
    dices = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        p = (pred == c)
        g = (gt   == c)
        inter = 2.0 * np.sum(p & g)
        denom = np.sum(p) + np.sum(g) + eps
        dices[c] = inter / denom
    return dices

def keep_largest_cc(mask):
    """mask: (D,H,W) uint8 二值 -> 保留最大连通域"""
    try:
        from scipy.ndimage import label
    except Exception:
        return mask
    cc, n = label(mask)
    if n <= 1:
        return mask
    sizes = [(cc == i).sum() for i in range(1, n+1)]
    keep = 1 + int(np.argmax(sizes))
    return (cc == keep).astype(np.uint8)

def edges2(rgb_gray, mask):
    """简易边缘覆盖到灰度图上"""
    img = (rgb_gray - rgb_gray.min()) / (rgb_gray.max() - rgb_gray.min() + 1e-6)
    edge = (mask != np.roll(mask, 1, 0)) | (mask != np.roll(mask, 1, 1))
    rgb = np.stack([img, img, img], axis=-1)
    rgb[edge] = [1, 0, 0]
    return rgb

def save_grid(vol_d, pred_d, out_path, ks):
    D = vol_d.shape[0]
    n = len(ks)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i, k in enumerate(ks):
        ax = plt.subplot(rows, cols, i+1)
        k = int(np.clip(k, 0, D-1))
        ax.imshow(edges2(vol_d[k], pred_d[k]))
        ax.set_title(f"k={k}")
        ax.axis('off')
    plt.tight_layout(); plt.savefig(out_path); plt.close()


# ------------------ DEBUG 工具 ------------------
def summarize_volume(vol, name, out_dir):
    """导出体数据概览 & 直方图"""
    os.makedirs(out_dir, exist_ok=True)
    info = {
        "name": name,
        "shape": tuple(int(x) for x in vol.shape),
        "dtype": str(vol.dtype),
        "min": float(np.min(vol)),
        "max": float(np.max(vol)),
        "mean": float(np.mean(vol)),
        "std": float(np.std(vol) + 1e-8),
    }
    with open(os.path.join(out_dir, f"_debug_{name}_stats.json"), "w") as f:
        json.dump(info, f, indent=2)

    # 直方图
    try:
        plt.figure(figsize=(4,3))
        v = vol[~np.isnan(vol)]
        v = np.clip(v, np.percentile(v, 0.5), np.percentile(v, 99.5))
        plt.hist(v.flatten(), bins=80)
        plt.title(f"{name} hist"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"_debug_{name}_hist.png")); plt.close()
    except Exception:
        pass
    return info

def bbox_3d(mask):
    """返回非零体素包围盒 (z0,z1,y0,y1,x0,x1)，若全 0 返回 None"""
    idx = np.where(mask > 0)
    if len(idx[0]) == 0: return None
    z0, z1 = int(np.min(idx[0])), int(np.max(idx[0])) + 1
    y0, y1 = int(np.min(idx[1])), int(np.max(idx[1])) + 1
    x0, x1 = int(np.min(idx[2])), int(np.max(idx[2])) + 1
    return (z0, z1, y0, y1, x0, x1)

def save_overlay(vol, mask, prob=None, out_path=None, k=None, title=""):
    """单切片覆盖：原图 + 预测边缘 + (可选)概率热力图"""
    D = vol.shape[0]
    k = D//2 if k is None else int(np.clip(k, 0, D-1))
    fig, ax = plt.subplots(1, 2 if prob is None else 3, figsize=(10,4))
    ax0 = ax[0] if isinstance(ax, (list, np.ndarray)) else ax
    ax0.imshow(edges2(vol[k], mask[k])); ax0.set_title(f"overlay k={k}"); ax0.axis('off')
    if prob is not None:
        ax1 = ax[1]; ax1.imshow(vol[k], cmap='gray'); im=ax1.imshow(prob[k], cmap='hot', alpha=0.5)
        ax1.set_title("prob heat"); ax1.axis('off'); plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax2 = ax[2]; ax2.imshow(edges2(vol[k], (prob[k] > 0.5).astype(np.uint8)))
        ax2.set_title("prob>0.5 edge"); ax2.axis('off')
    if title: fig.suptitle(title)
    plt.tight_layout()
    if out_path: plt.savefig(out_path)
    plt.close()

def affine_summary(nii):
    """简要打印/记录 affine（用于排查配准/朝向）"""
    A = np.asarray(nii.affine)
    vox = np.sqrt((A[:3,:3]**2).sum(0))  # 近似体素间距
    return {"affine": A.tolist(), "voxel_size_approx": [float(x) for x in vox]}


# ------------------ 主逻辑 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='3D NIfTI (.nii/.nii.gz)')
    ap.add_argument('--weights', required=True, help='best.pt')
    ap.add_argument('--classes', type=int, default=None, help='类别数；留空则从权重 meta 自动读取')
    ap.add_argument('--out_dir', default='./pred3d_vis')

    # 可视化
    ap.add_argument('--slices', default='', help='切片索引，40,60,80')
    ap.add_argument('--grid', action='store_true', help='保存切片蒙太奇')
    # 保存
    ap.add_argument('--save_nii', action='store_true', help='保存预测 .nii.gz')
    ap.add_argument('--prob', action='store_true', help='保存概率体积 .nii.gz')
    ap.add_argument('--prob_class', type=int, default=None, help='概率体的类别ID；默认=前列腺ID或1(二分类)')

    # 二分类模式
    ap.add_argument('--binary_prostate', action='store_true', help='只保留前列腺(=1)，其他置0')
    ap.add_argument('--prostate_label', type=int, default=5, help='多类标签中的前列腺ID（默认5）')

    # 评估
    ap.add_argument('--label', default='', help='标签 NIfTI 路径；为空则尝试自动映射')
    ap.add_argument('--label_root', default='', help='自动映射时标签的根目录')
    ap.add_argument('--label_dir', default='semantic_labels_only', help='标签子目录名')
    ap.add_argument('--img_token', default='LFOV'); ap.add_argument('--lab_token', default='SEMANTIC')
    ap.add_argument('--metrics_csv', action='store_true', help='把指标写到 CSV')

    # 推理增强/后处理
    ap.add_argument('--tta', action='store_true', help='x/y/z 轴翻转集成')
    ap.add_argument('--postprocess_lcc', action='store_true', help='最大连通域后处理（二分类或仅前列腺）')

    # DEBUG / 额外选项
    ap.add_argument('--debug', action='store_true', help='输出体检报告与中间图像')
    ap.add_argument('--debug_thresholds', default='0.2,0.3,0.4,0.5', help='阈值评估，逗号分隔')
    ap.add_argument('--resize_dhw', default='', help='例如 128,256,256；留空不缩放')
    ap.add_argument('--force_sigmoid', action='store_true', help='强制走1通道sigmoid推理（二分类权重）')
    ap.add_argument('--thr', type=float, default=0.35, help='二分类sigmoid阈值（仅在sigmoid模式）')

    ap.add_argument('--softmax_thr', type=float, default=-1.0,
                help='二分类softmax时用概率阈值而不是argmax；>=0生效，例如 0.3')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 读图 & 预处理 ----
    img_nii = nib.load(args.image)
    vol_xyz = img_nii.get_fdata(caching='unchanged').astype(np.float32)
    vol = to_dhw(vol_xyz)

    # 裁剪 + 非零体素 z-score（更贴近训练常用设置）
    mask_nz = vol > 0
    if mask_nz.any():
        p1, p99 = np.percentile(vol[mask_nz], [0.5, 99.5])
    else:
        p1, p99 = np.percentile(vol, [0.5, 99.5])
    vol = np.clip(vol, p1, p99)

    if mask_nz.any():
        m = float(vol[mask_nz].mean()); s = float(vol[mask_nz].std() + 1e-6)
    else:
        m = float(vol.mean()); s = float(vol.std() + 1e-6)
    vol_n = (vol - m) / s
    vol_n = vol_n.astype(np.float32, copy=False)

    # 可选：几何缩放到固定 (D,H,W)
    if args.resize_dhw.strip():
        target = [int(v) for v in args.resize_dhw.split(',')]
        xt = torch.from_numpy(vol_n[None, None].astype(np.float32))  # (1,1,D,H,W)
        xt = F.interpolate(xt, size=tuple(target), mode='trilinear', align_corners=False)
        vol_n = xt.numpy().squeeze(0).squeeze(0)

    x = torch.from_numpy(vol_n).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # ---- 权重 & 类别数 ----
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    meta = ckpt.get('meta', {})
    eff_classes = meta.get('effective_classes', meta.get('classes'))
    n_classes = args.classes if args.classes is not None else (eff_classes if eff_classes is not None else 6)

    # 从权重里猜“最后一层通道数”
    final_w = ckpt['model'].get('final_conv.weight', None)
    head_out_ch = int(final_w.shape[0]) if final_w is not None else n_classes
    print(f"[Info] head_out_ch(from_ckpt)={head_out_ch}, n_classes(param)={n_classes}")

    model = UNet3D_Improved(in_ch=1, n_classes=n_classes).to(device).eval()
    model.load_state_dict(ckpt['model'])
    model = model.to(device).float()

    # 判定是否走“sigmoid 单通道二分类”路径
    sigmoid_mode = args.force_sigmoid or (head_out_ch == 1) or (n_classes == 1)

    # ---- 推理（TTA 可选）----
    with torch.no_grad():
        if not args.tta:
            logits = model(x)  # (1,C,D,H,W)
            if sigmoid_mode:
                probs = torch.sigmoid(logits).cpu()     # (1,1,D,H,W)
            else:
                probs = torch.softmax(logits, dim=1).cpu()
        else:
            flips = [(), (2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
            acc = None
            for f in flips:
                xf = torch.flip(x, dims=f) if f else x
                lo = model(xf)
                pf = torch.sigmoid(lo) if sigmoid_mode else torch.softmax(lo, dim=1)
                pf = torch.flip(pf, dims=f) if f else pf
                acc = pf if acc is None else (acc + pf)
            probs = (acc / len(flips)).cpu()

        if sigmoid_mode:
            prob_fg = probs[0, 0].numpy()
            pred_bin = (prob_fg > args.thr).astype(np.uint8)
            pred_mc = pred_bin.copy()
        else:
            # 二分类softmax：可选阈值二值化
            if (probs.shape[1] == 2) and (args.softmax_thr >= 0):
                prob_fg = probs[0, 1].numpy()                # 通道1是前景
                pred_bin = (prob_fg > args.softmax_thr).astype(np.uint8)
                pred_mc = pred_bin.copy()                    # 下游统一走 pred_mc
            else:
                pred_mc = torch.argmax(probs, dim=1).squeeze(0).numpy()

    # ---- 二分类/后处理 ----
    if args.binary_prostate:
        if sigmoid_mode:
            pred = pred_mc.astype(np.uint8)  # 已经是0/1
        else:
            prostate_id = args.prostate_label if n_classes > 2 else 1
            pred = (pred_mc == prostate_id).astype(np.uint8)
        if args.postprocess_lcc:
            pred = keep_largest_cc(pred)
    else:
        pred = pred_mc.copy()
        if args.postprocess_lcc and (not sigmoid_mode):
            prostate_id = args.prostate_label if n_classes > 2 else 1
            mask = (pred_mc == prostate_id).astype(np.uint8)
            mask = keep_largest_cc(mask)
            pred = np.where(mask == 1, prostate_id, pred_mc)

    # ---- 可视化 ----
    stem = os.path.basename(args.image).replace('.nii', '').replace('.gz', '')
    ks = [int(k) for k in args.slices.split(',') if k.strip().isdigit()]
    if not ks:
        D = vol_n.shape[0]
        ks = [D//3, D//2, (2*D)//3]

    if args.grid:
        save_grid(vol_n, pred, os.path.join(args.out_dir, f'{stem}_grid.png'), ks)
    else:
        for k in ks:
            k = int(np.clip(k, 0, vol_n.shape[0]-1))
            plt.imsave(os.path.join(args.out_dir, f'{stem}_k{k}.png'),
                       edges2(vol_n[k], pred[k]))

    # ---- 保存 NIfTI / 概率 ----
    if args.save_nii:
        pred_xyz = from_dhw(pred.astype(np.int16))
        nib.save(nib.Nifti1Image(pred_xyz, affine=img_nii.affine, header=img_nii.header),
                 os.path.join(args.out_dir, f'{stem}_pred.nii.gz'))

    if args.prob:
        if sigmoid_mode:
            prob_vol = probs[0, 0].numpy()
        else:
            pc = args.prob_class
            if pc is None:
                pc = args.prostate_label if (not args.binary_prostate and n_classes > 2) else 1
            pc = int(pc)
            prob_vol = probs[0, pc].numpy()
        prob_xyz = from_dhw(prob_vol.astype(np.float32))
        nib.save(nib.Nifti1Image(prob_xyz, affine=img_nii.affine, header=img_nii.header),
                 os.path.join(args.out_dir, f'{stem}_prob.nii.gz'))

    # ---- 评估（如果给了标签）----
    gt_path = args.label.strip()
    if gt_path == '' and args.label_root != '':
        lab_base = default_name_mapper(os.path.basename(args.image), args.img_token, args.lab_token)
        gt_path = os.path.join(args.label_root, args.label_dir, lab_base)

    result = {
        'weights': os.path.abspath(args.weights),
        'image': os.path.abspath(args.image),
        'effective_classes': int(n_classes),
        'binary_mode': bool(args.binary_prostate),
        'sigmoid_mode': bool(sigmoid_mode),
        'prostate_label': int(args.prostate_label),
        'slices': ks,
        'out_dir': os.path.abspath(args.out_dir),
    }

    dice_macro_to_csv = None
    if gt_path and os.path.exists(gt_path):
        gt_nii = nib.load(gt_path)
        gt_xyz = gt_nii.get_fdata(caching='unchanged')
        if gt_xyz.ndim == 4: gt_xyz = gt_xyz[..., 0]
        gt = to_dhw(gt_xyz).astype(np.int64)

        if gt.shape != pred.shape:
            result['shape_mismatch'] = {'pred': tuple(int(x) for x in pred.shape),
                                        'gt':   tuple(int(x) for x in gt.shape)}
            print("[WARN] GT shape != PRED shape:", result['shape_mismatch'])

        if args.binary_prostate:
            pros_id = args.prostate_label  # 一律按数据集真实标签值来二值化GT（例如5）
            gt_bin = (gt == pros_id).astype(np.uint8)

            # Dice@当前预测
            num = 2.0 * np.sum((pred == 1) & (gt_bin == 1))
            den = np.sum(pred == 1) + np.sum(gt_bin == 1) + 1e-6
            dice_bin = float(num / den)
            result['dice_binary_prostate'] = dice_bin
            print(f"[Metrics] Binary prostate Dice = {dice_bin:.4f}")

            # 阈值扫描（sigmoid模式有意义；softmax模式也可看一下）
            try:
                thresholds = [float(t) for t in args.debug_thresholds.split(',') if t.strip() != '']
                debug_dice = {}
                if sigmoid_mode:
                    prob = probs[0, 0].numpy()
                else:
                    pc = 1 if n_classes == 2 else args.prostate_label
                    prob = probs[0, int(pc)].numpy()
                for t in thresholds:
                    m = (prob > t).astype(np.uint8)
                    if args.postprocess_lcc: m = keep_largest_cc(m)
                    num = 2.0 * np.sum((m == 1) & (gt_bin == 1))
                    den = np.sum(m == 1) + np.sum(gt_bin == 1) + 1e-6
                    debug_dice[f"thr@{t}"] = float(num/den)
                    save_overlay(vol_n, m, prob=None,
                                 out_path=os.path.join(args.out_dir, f"{stem}_debug_thr_{t}.png"),
                                 title=f"thr={t}")
                result['dice_threshold_sweep'] = debug_dice
                print("[Debug] Dice by thresholds:", debug_dice)
            except Exception:
                pass

            # 体素数/包围盒
            result['gt_positive_voxels']   = int(gt_bin.sum())
            result['pred_positive_voxels'] = int((pred==1).sum())
            result['gt_bbox']   = bbox_3d(gt_bin)
            result['pred_bbox'] = bbox_3d(pred)

            dice_macro_to_csv = dice_bin

        else:
            dices = dice_per_class_np(pred, gt, num_classes=n_classes)
            macro = float(np.mean(dices))
            pros_id = args.prostate_label if n_classes > 2 else 1
            result['dice_macro'] = macro
            result['dice_per_class'] = {f'c{i}': float(d) for i, d in enumerate(dices)}
            result['dice_prostate'] = float(dices[pros_id])
            print(f"[Metrics] macro={macro:.4f} | prostate={dices[pros_id]:.4f}")
            dice_macro_to_csv = macro
    else:
        print("[Metrics] Ground-truth not provided; skip metric computation.")
        result['dice_macro'] = None

    # ---- DEBUG 汇总 ----
    if args.debug:
        result['image_affine'] = affine_summary(img_nii)
        summarize_volume(from_dhw(vol), 'image_raw_xyz', args.out_dir)
        summarize_volume(vol_n, 'image_dhw', args.out_dir)
        if args.prob:
            pv = probs[0, 0].numpy() if sigmoid_mode else probs[0, (1 if n_classes==2 else args.prostate_label)].numpy()
            summarize_volume(pv, 'prob_dhw', args.out_dir)
            save_overlay(vol_n, pred if args.binary_prostate else (pred==(args.prostate_label if n_classes>2 else 1)).astype(np.uint8),
                         prob=pv, out_path=os.path.join(args.out_dir, f"{stem}_debug_prob.png"))

    # ---- 保存 meta ----
    with open(os.path.join(args.out_dir, f'{stem}_meta.json'), 'w') as f:
        json.dump(result, f, indent=2)

    if args.metrics_csv and (dice_macro_to_csv is not None):
        csv_path = os.path.join(args.out_dir, 'metrics.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if args.binary_prostate:
                header = ['image', 'mode', 'dice_binary_prostate']
                row = [stem, 'binary', dice_macro_to_csv]
            else:
                header = ['image', 'mode', 'dice_macro', 'dice_prostate'] + [f'c{i}' for i in range(n_classes)]
                row = [stem, 'multiclass', dice_macro_to_csv, result['dice_prostate']] + \
                      [result['dice_per_class'][f'c{i}'] for i in range(n_classes)]
            if write_header: writer.writerow(header)
            writer.writerow(row)

    print(f"Saved visualizations to: {args.out_dir}")
    if args.save_nii: print("  + prediction NIfTI saved")
    if args.prob:     print("  + probability NIfTI saved")
    if gt_path and os.path.exists(gt_path): print(f"  + metrics computed against: {gt_path}")


if __name__ == '__main__':
    main()
