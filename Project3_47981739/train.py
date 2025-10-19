import os, argparse, math, time
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
import matplotlib.pyplot as plt

from dataset import HipMRI2DNIIDataset
from modules import UNet2D_Improved, DiceLoss, dice_score

def build_loaders(data_root, batch=8, val_ratio=0.1, workers=4, augment=True, classes=5):
    ds = HipMRI2DNIIDataset(data_root, augment=augment, classes=classes)
    n_val = max(1, int(len(ds) * val_ratio))
    n_tr  = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=True)
    va_loader = DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return tr_loader, va_loader

def train_one_epoch(model, loader, opt, scaler, device, criterion):
    model.train()
    run_loss, run_dice, n = 0.0, 0.0, 0
    for img, mask, _ in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
            logits = model(img)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            run_dice += dice_score(probs, mask).item() * img.size(0)
            run_loss += loss.item() * img.size(0)
            n += img.size(0)
    return run_loss / n, run_dice / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_dice, tot = 0.0, 0
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        tot_dice += dice_score(probs, mask).item() * img.size(0)
        tot += img.size(0)
    return tot_dice / tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='root containing images/ and labels/')
    ap.add_argument('--out_dir', default='./runs/hipmri2d_unet')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--classes', type=int, default=5)
    ap.add_argument('--base', type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_loader, va_loader = build_loaders(args.data_root, batch=args.batch, classes=args.classes)
    model = UNet2D_Improved(in_ch=1, n_classes=args.classes, base=args.base).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    criterion = DiceLoss()

    best_val = -1.0
    hist = {'tr_loss':[], 'tr_dice':[], 'va_dice':[]}

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_dice = train_one_epoch(model, tr_loader, opt, scaler, device, criterion)
        va_dice = evaluate(model, va_loader, device)
        t1 = time.time()
        hist['tr_loss'].append(tr_loss); hist['tr_dice'].append(tr_dice); hist['va_dice'].append(va_dice)
        print(f'Epoch {ep:03d}/{args.epochs}  loss={tr_loss:.4f}  trDice={tr_dice:.4f}  valDice={va_dice:.4f}  ({t1-t0:.1f}s)')

        if va_dice > best_val:
            best_val = va_dice
            torch.save({'model': model.state_dict(),
                        'meta': vars(args),
                        'val_dice': va_dice}, os.path.join(args.out_dir, 'best.pt'))
            print(f'  ✓ saved best.pt (valDice={va_dice:.4f})')

    # plot curves
    try:
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot(hist['tr_loss']); plt.title('Train Loss'); plt.xlabel('epoch'); plt.ylabel('loss')
        plt.savefig(os.path.join(args.out_dir, 'loss.png')); plt.close()
        plt.figure(); plt.plot(hist['tr_dice'], label='train'); plt.plot(hist['va_dice'], label='val'); plt.legend()
        plt.title('Dice'); plt.xlabel('epoch'); plt.ylabel('Dice'); plt.savefig(os.path.join(args.out_dir,'dice.png')); plt.close()
    except Exception as e:
        print('plot failed:', e)

if __name__ == '__main__':
    main()
