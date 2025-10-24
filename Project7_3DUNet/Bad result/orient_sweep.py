# orient_sweep.py — 轴映射/翻转穷举排错
import os, argparse, itertools, numpy as np, nibabel as nib, torch, torch.nn.functional as F
from modules import UNet3D_Improved

def to_perm(vol_xyz, perm):
    # perm 是 (d,h,w) 各自来自 xyz 的索引，例如 (2,1,0) 表示 D=Z(2), H=Y(1), W=X(0)
    return np.transpose(vol_xyz, perm)

def from_dhw(vol_dhw):  # (D,H,W)->(X,Y,Z) 仅用于保存nii
    return np.transpose(vol_dhw, (2,1,0))

def bin_dice(pred, gt, eps=1e-6):
    num = 2.0 * np.sum((pred==1) & (gt==1))
    den = np.sum(pred==1) + np.sum(gt==1) + eps
    return float(num/den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--prostate_label', type=int, default=5)
    ap.add_argument('--out_dir', default='./orient_sweep_out')
    ap.add_argument('--resize_dhw', default='128,256,256')
    ap.add_argument('--thr', type=float, default=0.35)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_nii = nib.load(args.image)
    lab_nii = nib.load(args.label)
    vol_xyz = img_nii.get_fdata(caching='unchanged').astype(np.float32)
    lab_xyz = lab_nii.get_fdata(caching='unchanged')
    if vol_xyz.ndim==4: vol_xyz=vol_xyz[...,0]
    if lab_xyz.ndim==4: lab_xyz=lab_xyz[...,0]
    lab_xyz = lab_xyz.astype(np.int64)

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    n_classes = int(ckpt.get('meta',{}).get('effective_classes', 2))
    head_out = ckpt['model'].get('final_conv.weight', None)
    head_ch  = int(head_out.shape[0]) if head_out is not None else n_classes
    sigmoid_mode = (head_ch==1) or (n_classes==1)

    model = UNet3D_Improved(in_ch=1, n_classes=n_classes).to(device).eval()
    model.load_state_dict(ckpt['model'])
    model.float()

    perms = list(itertools.permutations([0,1,2], 3))  # 6 种 xyz->dhw
    flips = [()] + [(i,) for i in [0,1,2]] + [(0,1),(0,2),(1,2),(0,1,2)]  # 对 (D,H,W) 轴翻转的8种组合

    best = {'dice': -1.0}
    for perm in perms:
        # 图像预处理（和你现在推理一致：裁剪 + 非零 zscore + resize）
        vol = to_perm(vol_xyz, (perm[2], perm[1], perm[0]))  # 注意：我们要把 (X,Y,Z) 映射到 (D,H,W)=(?, ?, ?)
        lab = to_perm(lab_xyz, (perm[2], perm[1], perm[0]))
        mask_nz = vol > 0
        p1,p99 = (np.percentile(vol[mask_nz],[0.5,99.5]) if mask_nz.any()
                  else np.percentile(vol,[0.5,99.5]))
        vol = np.clip(vol, p1, p99)
        if mask_nz.any():
            m,s = float(vol[mask_nz].mean()), float(vol[mask_nz].std()+1e-6)
        else:
            m,s = float(vol.mean()), float(vol.std()+1e-6)
        vol = (vol - m) / s
        # resize
        D,H,W = [int(v) for v in args.resize_dhw.split(',')]
        xt = torch.from_numpy(vol[None,None].astype(np.float32))
        xt = F.interpolate(xt, size=(D,H,W), mode='trilinear', align_corners=False)
        vol = xt.numpy().squeeze(0).squeeze(0)
        # label 同样 resize 最近邻
        yt = torch.from_numpy(lab[None,None].astype(np.float32))
        yt = F.interpolate(yt, size=(D,H,W), mode='nearest')
        lab = yt.numpy().squeeze(0).squeeze(0).astype(np.int64)
        lab_bin = (lab == (1 if n_classes==2 else args.prostate_label)).astype(np.uint8)

        x = torch.from_numpy(vol).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            lo = model(x)
            if sigmoid_mode:
                pr = torch.sigmoid(lo).cpu()[0,0].numpy()
            else:
                pr_all = torch.softmax(lo, dim=1).cpu()[0].numpy()
                pr = pr_all[1 if n_classes==2 else args.prostate_label]

        # 对每个 flip 也试一下
        for f in flips:
            prf = pr.copy()
            lbf = lab_bin.copy()
            if f:
                prf = np.flip(prf, axis=f)
                lbf = np.flip(lbf, axis=f)
            pred_bin = (prf > args.thr).astype(np.uint8)
            d = bin_dice(pred_bin, lbf)
            if d > best['dice']:
                best = {'dice': d, 'perm': perm, 'flip': f}

    print("=== BEST ORIENTATION ===")
    print("perm xyz->dhw  (D,H,W) takes from indices:", best['perm'], "(0=X,1=Y,2=Z)")
    print("flips on (D,H,W) axes:", best['flip'])
    print("max Dice @ thr={}: {:.4f}".format(args.thr, best['dice']))

    # 可选：把这一套用于保存一个预测 NIfTI
    # （省略；我们只关心找到 perm/flip）
if __name__ == "__main__":
    main()
