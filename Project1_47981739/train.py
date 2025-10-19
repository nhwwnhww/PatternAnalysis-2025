# train.py
import os, argparse, inspect
import torch
from torch import amp as torch_amp
from torch.utils.data import DataLoader
from dataset import OASIS2DPNGDataset
from modules import UNet2D, dice_loss, dice_score

def get_loaders(data_root, img_size, num_classes, batch_size, num_workers):
    train_ds = OASIS2DPNGDataset(os.path.join(data_root, "keras_png_slices_train"),
                                 os.path.join(data_root, "keras_png_slices_seg_train"),
                                 img_size=img_size, augment=True, num_classes=num_classes)
    val_ds   = OASIS2DPNGDataset(os.path.join(data_root, "keras_png_slices_validate"),
                                 os.path.join(data_root, "keras_png_slices_seg_validate"),
                                 img_size=img_size, augment=False, num_classes=num_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    return train_ds, val_ds, train_loader, val_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--save_dir",  type=str, default="./runs_oasis_unet")
    ap.add_argument("--epochs",    type=int, default=20)
    ap.add_argument("--img_size",  type=int, default=128)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP_DTYPE = torch.float16

    _, val_ds, train_loader, val_loader = get_loaders(
        args.data_root, args.img_size, args.num_classes, args.batch_size, args.num_workers
    )

    model = UNet2D(in_c=1, n_classes=args.num_classes, base=32).to(device)
    model = model.to(memory_format=torch.channels_last)

    opt_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if "foreach" in inspect.signature(torch.optim.AdamW).parameters:
        opt_kwargs["foreach"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    scaler = torch_amp.GradScaler(enabled=(device.type=="cuda"))

    best_val = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs  = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch_amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(device.type=="cuda")):
                logits = model(imgs)
                loss = dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                masks = masks.to(device, non_blocking=True)
                with torch_amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(device.type=="cuda")):
                    logits = model(imgs)
                    batch_loss = dice_loss(logits, masks).item()
                val_loss += batch_loss * imgs.size(0)
                val_dice += dice_score(logits, masks) * imgs.size(0)
        val_loss /= len(val_ds); val_dice /= len(val_ds)

        print(f"Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValDice {val_dice:.4f}")

        if val_dice > best_val:
            best_val = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": best_val}, best_path)
            print("  ✓ Saved best:", best_path)

if __name__ == "__main__":
    main()
