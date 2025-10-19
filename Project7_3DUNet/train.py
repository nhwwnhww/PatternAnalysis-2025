# train.py
import os, argparse, time, csv
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch import amp as torch_amp
from tqdm import tqdm
import numpy as np

from dataset import HipMRI3DNIIDataset
from modules import UNet3D_Improved, DiceLoss3D, dice_score_3d

@torch.no_grad()
def dice_per_class(probs, target, eps=1e-6):
    N, C = probs.shape[:2]
    oh = torch.nn.functional.one_hot(target.long(), num_classes=C).permute(0,4,1,2,3).float()
    num = 2 * (probs * oh).sum(dim=(0,2,3,4))
    den = (probs + oh).sum(dim=(0,2,3,4)).clamp_min(eps)
    return (num / den).detach().cpu().numpy()  # (C,)

def build_loaders(root, batch=1, val_ratio=0.1, workers=4, patch="96,160,160",
                  classes=5, augment=True, image_dir='images', label_dir='labels',
                  binary_prostate=False, prostate_label=5):
    patch = tuple(int(x) for x in patch.split(','))
    ds = HipMRI3DNIIDataset(root,
                            image_dir=image_dir,
                            label_dir=label_dir,
                            patch=patch,
                            augment=augment,
                            classes=classes,
                            img_token='LFOV',
                            lab_token='SEMANTIC',
                            binary_prostate=binary_prostate,
                            prostate_label=prostate_label)
    n_val = max(1, int(len(ds)*val_ratio)); n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True)
    va_loader = DataLoader(va, batch_size=1,    shuffle=False, num_workers=workers, pin_memory=True)
    return tr_loader, va_loader

@torch.no_grad()
def evaluate(model, loader, device, want_classwise=False):
    model.eval()
    tot_macro, n = 0.0, 0
    acc_cls = None
    for img, lab, _ in loader:
        img, lab = img.to(device), lab.to(device)
        logits = model(img)
        probs  = torch.softmax(logits, dim=1)
        tot_macro += dice_score_3d(probs, lab).item()
        if want_classwise:
            dpc = dice_per_class(probs, lab)
            acc_cls = dpc if acc_cls is None else (acc_cls + dpc)
        n += 1
    macro = tot_macro / max(1, n)
    classwise = (acc_cls / max(1, n)) if want_classwise and acc_cls is not None else None
    return macro, classwise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--image_dir', default='images')
    ap.add_argument('--label_dir', default='labels')
    ap.add_argument('--out_dir',   default='./runs/hipmri3d_unet')
    ap.add_argument('--epochs',    type=int, default=200)
    ap.add_argument('--batch',     type=int, default=1)
    ap.add_argument('--lr',        type=float, default=2e-4)
    ap.add_argument('--base',      type=int, default=16)
    ap.add_argument('--classes',   type=int, default=6)  # 多类默认 6（BG+5）
    ap.add_argument('--patch',     default='96,160,160')
    ap.add_argument('--workers',   type=int, default=4)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--no_aug',    action='store_true')
    # 二分类相关
    ap.add_argument('--binary_prostate', action='store_true',
                    help='将标签二值化 (prostate=1, others=0)，并自动把 classes 设为 2')
    ap.add_argument('--prostate_label', type=int, default=5,
                    help='原始标签中前列腺的类别ID（默认 5）')
    ap.add_argument('--classwise', action='store_true', help='输出按类别 Dice')
    args = ap.parse_args()

    # 如果是二分类，强制 classes=2
    n_classes = 2 if args.binary_prostate else args.classes

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} | classes={n_classes} | binary={args.binary_prostate}')

    tr_loader, va_loader = build_loaders(root=args.data_root,
                                         batch=args.batch,
                                         val_ratio=args.val_ratio,
                                         workers=args.workers,
                                         patch=args.patch,
                                         classes=n_classes,
                                         augment=not args.no_aug,
                                         image_dir=args.image_dir,
                                         label_dir=args.label_dir,
                                         binary_prostate=args.binary_prostate,
                                         prostate_label=args.prostate_label)

    model  = UNet3D_Improved(in_ch=1, n_classes=n_classes, base=args.base).to(device)
    opt    = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch_amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    criterion = DiceLoss3D()

    best = -1.0
    hist = {'tr_loss': [], 'tr_dice': [], 'va_dice': []}

    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        header = ['epoch', 'train_loss', 'train_dice', 'val_dice']
        if args.classwise:
            header += [f'val_dice_c{i}' for i in range(n_classes)]
        writer.writerow(header)

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        run_loss = run_dice = 0.0
        steps = 0

        for img, lab, _ in tqdm(tr_loader, desc=f"Epoch {ep}/{args.epochs} [train]", ncols=110):
            img, lab = img.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
                logits = model(img)
                loss   = criterion(logits, lab)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                run_loss += loss.item()
                run_dice += dice_score_3d(probs, lab).item()
                steps += 1

        tr_loss = run_loss / max(1, steps)
        tr_dice = run_dice / max(1, steps)

        # 验证
        val_macro, val_classwise = evaluate(model, va_loader, device, want_classwise=args.classwise)
        dt = time.time() - t0
        print(f"Epoch {ep:03d}/{args.epochs} | loss={tr_loss:.4f}  trDice={tr_dice:.4f}  "
              f"valDice={val_macro:.4f} | {dt:.1f}s")

        hist['tr_loss'].append(tr_loss); hist['tr_dice'].append(tr_dice); hist['va_dice'].append(val_macro)

        # 记录 CSV
        with open(csv_path, 'a', newline='') as f:
            import csv
            writer = csv.writer(f)
            row = [ep, tr_loss, tr_dice, val_macro]
            if args.classwise and val_classwise is not None:
                row += list(np.asarray(val_classwise, dtype=float))
            writer.writerow(row)

        # 保存 best
        if val_macro > best:
            best = val_macro
            torch.save({'model': model.state_dict(),
                        'meta': {**vars(args), 'effective_classes': n_classes},
                        'val_dice': val_macro},
                       os.path.join(args.out_dir, 'best.pt'))
            print(f"  ✓ saved best.pt (valDice={val_macro:.4f})")

    # 画图
    try:
        import matplotlib.pyplot as plt
        x = np.arange(len(hist['tr_loss']))
        plt.figure(); plt.plot(x, hist['tr_loss']); plt.title('Train Loss'); plt.xlabel('epoch'); plt.ylabel('loss')
        plt.savefig(os.path.join(args.out_dir, 'loss.png')); plt.close()
        plt.figure(); plt.plot(x, hist['tr_dice'], label='train'); plt.plot(x, hist['va_dice'], label='val'); plt.legend()
        plt.title('Dice'); plt.xlabel('epoch'); plt.ylabel('Dice')
        plt.savefig(os.path.join(args.out_dir, 'dice.png')); plt.close()
    except Exception as e:
        print('plot failed:', e)

if __name__ == '__main__':
    main()
