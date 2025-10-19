import os, argparse, time
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import HipMRI2DNIIDataset
from modules import UNet2D_Improved, DiceLoss, dice_score

class DiceFocalLoss(torch.nn.Module):
    """Dice + Focal Loss for multi-class segmentation"""
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.dice = DiceLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # Dice part
        dice_loss = self.dice(logits, target)
        # Focal part
        probs = torch.softmax(logits, dim=1)
        target_onehot = torch.nn.functional.one_hot(target, num_classes=logits.shape[1]).permute(0,3,1,2)
        probs = torch.clamp(probs, 1e-6, 1-1e-6)
        focal_loss = - (target_onehot * (1 - probs) ** self.gamma * torch.log(probs)).sum(dim=1).mean()
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss

def build_loaders(data_root, batch=8, val_ratio=0.1, workers=4, augment=True, classes=5):
    ds = HipMRI2DNIIDataset(data_root, augment=augment, classes=classes)
    n_val = max(1, int(len(ds) * val_ratio))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    va_loader = DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return tr_loader, va_loader

def train_one_epoch(model, loader, opt, scaler, device, criterion, epoch=None, total_epochs=None):
    model.train()
    run_loss, run_dice, n = 0.0, 0.0, 0
    desc = f"Epoch [{epoch}/{total_epochs}] Training" if epoch else "Training"
    pbar = tqdm(loader, desc=desc, leave=False)
    for img, mask, _ in pbar:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        assert mask.max() < model.n_classes, f"Mask has class {mask.max()}, expected < {model.n_classes}"

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
            logits = model(img)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            batch_dice = dice_score(probs, mask).item()
            run_dice += batch_dice * img.size(0)
            run_loss += loss.item() * img.size(0)
            n += img.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}','dice': f'{batch_dice:.4f}'})
    return run_loss / n, run_dice / n

@torch.no_grad()
def evaluate(model, loader, device, epoch=None, total_epochs=None):
    model.eval()
    tot_dice, tot = 0.0, 0
    desc = f"Epoch [{epoch}/{total_epochs}] Validating" if epoch else "Validating"
    pbar = tqdm(loader, desc=desc, leave=False)
    for img, mask, _ in pbar:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        batch_dice = dice_score(probs, mask).item()
        tot_dice += batch_dice * img.size(0)
        tot += img.size(0)
        pbar.set_postfix({'dice': f'{batch_dice:.4f}'})
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
    scaler = torch.amp.GradScaler(enabled=(device.type=='cuda'))
    criterion = DiceFocalLoss(alpha=0.5)

    # Cosine LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_val = -1.0
    hist = {'tr_loss': [], 'tr_dice': [], 'va_dice': []}

    print(f"\n🚀 Starting training on device: {device}\n")

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_dice = train_one_epoch(model, tr_loader, opt, scaler, device, criterion, ep, args.epochs)
        va_dice = evaluate(model, va_loader, device, ep, args.epochs)
        scheduler.step()
        t1 = time.time()

        hist['tr_loss'].append(tr_loss)
        hist['tr_dice'].append(tr_dice)
        hist['va_dice'].append(va_dice)

        print(f"Epoch {ep:03d}/{args.epochs} | Loss={tr_loss:.4f} | TrainDice={tr_dice:.4f} | ValDice={va_dice:.4f} | Time={t1-t0:.1f}s")

        if va_dice > best_val:
            best_val = va_dice
            torch.save({'model': model.state_dict(), 'meta': vars(args), 'val_dice': va_dice},
                       os.path.join(args.out_dir, 'best.pt'))
            print(f"  ✅ Saved new best model (valDice={va_dice:.4f})")

    # Plot curves
    try:
        plt.figure()
        plt.plot(hist['tr_loss'])
        plt.title('Train Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.out_dir, 'loss.png'))
        plt.close()

        plt.figure()
        plt.plot(hist['tr_dice'], label='train')
        plt.plot(hist['va_dice'], label='val')
        plt.legend()
        plt.title('Dice')
        plt.xlabel('epoch')
        plt.ylabel('Dice')
        plt.savefig(os.path.join(args.out_dir, 'dice.png'))
        plt.close()
    except Exception as e:
        print('Plot failed:', e)

if __name__ == '__main__':
    main()

