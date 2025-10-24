# predict.py  (验收增强版)
import os, argparse, json, csv
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from modules import UNet3D_Improved

# ------------------ 工具函数 ------------------
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
    """pred/gt: (D,H,W) int; 返回 (num_classes,)"""
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
        # 没装 scipy 则直接返回
        return mask
    cc, n = label(mask)
    if n <= 1:
        return mask
    sizes = [(cc == i).sum() for i in range(1, n+1)]
    keep = 1 + int(np.argmax(sizes))
    return (cc == keep).astype(np.uint8)

def save_grid(vol_d, pred_d, out_path, ks):
    """把多张切片拼成一张图 (轴位)"""
    D = vol_d.shape[0]
    v = (vol_d - vol_d.min()) / (vol_d.max() - vol_d.min() + 1e-6)
    n = len(ks)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i, k in enumerate(ks):
        ax = plt.subplot(rows, cols, i+1)
        k = int(np.clip(k, 0, D-1))
        img = v[k]
        prm = pred_d[k]
        edge = (prm != np.roll(prm, 1, 0)) | (prm != np.roll(prm, 1, 1))
        rgb = np.stack([img, img, img], axis=-1)
        rgb[edge] = [1, 0, 0]
        ax.imshow(rgb)
        ax.set_title(f"k={k}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ------------------ 主逻辑 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='3D NIfTI 路径 (.nii/.nii.gz)')
    ap.add_argument('--weights', required=True, help='best.pt')
    ap.add_argument('--classes', type=int, default=None, help='类别数；留空则从权重 meta 自动读取')
    ap.add_argument('--out_dir', default='./pred3d_vis')

    # 可视化
    ap.add_argument('--slices', default='', help='自定义切片索引，逗号分隔，如 40,60,80')
    ap.add_argument('--grid', action='store_true', help='保存切片蒙太奇（多图拼成一张）')

    # NIfTI 保存
    ap.add_argument('--save_nii', action='store_true', help='保存预测 mask 为 .nii.gz')
    ap.add_argument('--prob', action='store_true', help='保存概率体积 .nii.gz')
    ap.add_argument('--prob_class', type=int, default=None, help='保存概率的类别ID；默认=前列腺ID或1(二分类时)')

    # 二分类模式
    ap.add_argument('--binary_prostate', action='store_true', help='只保留前列腺(=1)，其他置0')
    ap.add_argument('--prostate_label', type=int, default=5, help='前列腺在多类标签中的类别ID（默认5）')

    # 指标评估（提供标签任选一种方式）
    ap.add_argument('--label', default='', help='标签 NIfTI 路径；为空则尝试自动映射')
    ap.add_argument('--label_root', default='', help='如果用自动映射，标签所在根目录 (例如 semantic_labels_only 的上级目录)')
    ap.add_argument('--label_dir', default='semantic_labels_only', help='标签子目录名，用于自动映射')
    ap.add_argument('--img_token', default='LFOV')
    ap.add_argument('--lab_token', default='SEMANTIC')
    ap.add_argument('--metrics_csv', action='store_true', help='把指标写到 CSV 文件')

    # 推理增强和后处理
    ap.add_argument('--tta', action='store_true', help='翻转测试时增强 (x/y/z 轴翻转集成)')
    ap.add_argument('--postprocess_lcc', action='store_true', help='最大连通域后处理（二分类或针对前列腺）')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 读图 & 预处理 ----
    nii = nib.load(args.image)
    vol_xyz = nii.get_fdata(caching='unchanged').astype(np.float32)
    vol = to_dhw(vol_xyz)
    m, s = vol.mean(), vol.std() + 1e-6
    vol_n = (vol - m) / s
    x = torch.from_numpy(vol_n).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)

    # ---- 读权重 & 类别数 ----
    ckpt = torch.load(args.weights, map_location=device)
    meta = ckpt.get('meta', {})
    eff_classes = meta.get('effective_classes', meta.get('classes'))
    n_classes = args.classes if args.classes is not None else (eff_classes if eff_classes is not None else 6)

    model = UNet3D_Improved(in_ch=1, n_classes=n_classes).to(device).eval()
    model.load_state_dict(ckpt['model'])

    # ---- 推理（可选 TTA）----
    with torch.no_grad():
        if not args.tta:
            logits = model(x)                            # (1,C,D,H,W)
            probs = torch.softmax(logits, dim=1).cpu()
        else:
            # 8 种翻转组合（xyz 三轴开关），简单平均
            flips = [
                (), (2,), (3,), (4,),
                (2,3), (2,4), (3,4), (2,3,4)
            ]
            acc = None
            for f in flips:
                xf = x
                if f:
                    xf = torch.flip(xf, dims=(0,)+f)  # 0是batch，(2,3,4)分别对应 D/H/W
                pf = torch.softmax(model(xf), dim=1)
                if f:
                    pf = torch.flip(pf, dims=(0,)+f)
                acc = pf if acc is None else (acc + pf)
            probs = (acc / len(flips)).cpu()

        pred = torch.argmax(probs, dim=1).squeeze(0).numpy()  # (D,H,W)

    # ---- 二分类后处理（只保留前列腺）----
    if args.binary_prostate:
        prostate_id = args.prostate_label if n_classes > 2 else 1
        pred = (pred == prostate_id).astype(np.uint8)
        if args.postprocess_lcc:
            pred = keep_largest_cc(pred)
    elif args.postprocess_lcc:
        # 多类时，只对前列腺做 LCC，再“写回”前列腺通道
        prostate_id = args.prostate_label if n_classes > 2 else 1
        mask = (pred == prostate_id).astype(np.uint8)
        mask = keep_largest_cc(mask)
        pred = np.where(mask == 1, prostate_id, pred)

    # ---- 保存可视化（切片或拼图）----
    stem = os.path.basename(args.image).replace('.nii', '').replace('.gz', '')
    ks = [int(k) for k in args.slices.split(',') if k.strip().isdigit()]
    if not ks:
        D = vol.shape[0]
        ks = [D//3, D//2, 2*D//3]

    if args.grid:
        save_grid(vol, pred, os.path.join(args.out_dir, f'{stem}_grid.png'), ks)
    else:
        # 单张单存
        for k in ks:
            k = int(np.clip(k, 0, vol.shape[0]-1))
            img = (vol[k] - vol.min()) / (vol.max() - vol.min() + 1e-6)
            prm = pred[k]
            edge = (prm != np.roll(prm, 1, 0)) | (prm != np.roll(prm, 1, 1))
            rgb = np.stack([img, img, img], axis=-1)
            rgb[edge] = [1, 0, 0]
            plt.imsave(os.path.join(args.out_dir, f'{stem}_k{k}.png'), rgb)

    # ---- 保存 NIfTI ----
    if args.save_nii:
        pred_xyz = from_dhw(pred.astype(np.int16))
        nii_pred = nib.Nifti1Image(pred_xyz, affine=nii.affine, header=nii.header)
        nib.save(nii_pred, os.path.join(args.out_dir, f'{stem}_pred.nii.gz'))

    # 概率体积（前列腺或指定类）
    if args.prob:
        pc = args.prob_class
        if pc is None:
            pc = args.prostate_label if (not args.binary_prostate and n_classes > 2) else 1
        pc = int(pc)
        prob_vol = probs[0, pc].numpy()   # (D,H,W)
        prob_xyz = from_dhw(prob_vol.astype(np.float32))
        nii_prob = nib.Nifti1Image(prob_xyz, affine=nii.affine, header=nii.header)
        nib.save(nii_prob, os.path.join(args.out_dir, f'{stem}_prob_c{pc}.nii.gz'))

    # ---- 评估指标（如果给了标签或可自动映射到标签）----
    gt_path = args.label.strip()
    if gt_path == '' and args.label_root != '':
        img_base = os.path.basename(args.image)
        lab_base = default_name_mapper(img_base, args.img_token, args.lab_token)
        gt_path = os.path.join(args.label_root, args.label_dir, lab_base)

    result = {
        'weights': os.path.abspath(args.weights),
        'image': os.path.abspath(args.image),
        'effective_classes': int(n_classes),
        'binary_mode': bool(args.binary_prostate),
        'prostate_label': int(args.prostate_label),
        'slices': ks,
        'out_dir': os.path.abspath(args.out_dir),
    }

    if gt_path and os.path.exists(gt_path):
        gt_nii = nib.load(gt_path)
        gt_xyz = gt_nii.get_fdata(caching='unchanged')
        if gt_xyz.ndim == 4: gt_xyz = gt_xyz[..., 0]
        gt = to_dhw(gt_xyz).astype(np.int64)

        if args.binary_prostate:
            gt = (gt == (args.prostate_label if n_classes > 2 else 1)).astype(np.uint8)
            # pred 已经是 0/1
            num = 2.0 * np.sum((pred==1) & (gt==1))
            den = np.sum(pred==1) + np.sum(gt==1) + 1e-6
            dice_bin = float(num / den)
            result['dice_binary_prostate'] = dice_bin
            print(f"[Metrics] Binary prostate Dice = {dice_bin:.4f}")
        else:
            # 多类：报告 macro 与前列腺 Dice
            dices = dice_per_class_np(pred, gt, num_classes=n_classes)
            macro = float(np.mean(dices))
            pros_id = args.prostate_label if n_classes > 2 else 1
            result['dice_macro'] = macro
            result['dice_per_class'] = {f'c{i}': float(d) for i, d in enumerate(dices)}
            result['dice_prostate'] = float(dices[pros_id])
            print(f"[Metrics] macro={macro:.4f} | prostate={dices[pros_id]:.4f}")
    else:
        print("[Metrics] Ground-truth not provided; skip metric computation.")
        result['dice_macro'] = None

    # ---- 保存 meta ----
    with open(os.path.join(args.out_dir, f'{stem}_meta.json'), 'w') as f:
        json.dump(result, f, indent=2)

    if args.metrics_csv and result.get('dice_macro') is not None:
        csv_path = os.path.join(args.out_dir, 'metrics.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if args.binary_prostate:
                header = ['image', 'mode', 'dice_binary_prostate']
                row = [stem, 'binary', result['dice_binary_prostate']]
            else:
                header = ['image', 'mode', 'dice_macro', 'dice_prostate'] + [f'c{i}' for i in range(n_classes)]
                row = [stem, 'multiclass', result['dice_macro'], result['dice_prostate']] + \
                      [result['dice_per_class'][f'c{i}'] for i in range(n_classes)]
            if write_header: writer.writerow(header)
            writer.writerow(row)

    print(f"Saved visualizations to: {args.out_dir}")
    if args.save_nii: print("  + prediction NIfTI saved")
    if args.prob:     print("  + probability NIfTI saved")
    if gt_path and os.path.exists(gt_path): print(f"  + metrics computed against: {gt_path}")

if __name__ == '__main__':
    main()
