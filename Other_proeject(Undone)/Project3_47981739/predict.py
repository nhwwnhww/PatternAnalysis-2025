import os, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import HipMRI2DNIIDataset
from modules import UNet2D_Improved

def overlay(img, mask):
    """Quick visualization overlay: grayscale img + colored mask edges."""
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    edge = (mask != torch.roll(mask, 1, 0)) | (mask != torch.roll(mask, 1, 1))
    rgb = np.stack([img.squeeze().cpu().numpy()]*3, axis=-1)
    rgb[edge.cpu().numpy()] = [1,0,0]
    return rgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out_dir', default='./pred_vis')
    ap.add_argument('--classes', type=int, default=5)
    ap.add_argument('--num', type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = HipMRI2DNIIDataset(args.data_root, augment=False, classes=args.classes)
    model = UNet2D_Improved(in_ch=1, n_classes=args.classes).to(device).eval()
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'])

    with torch.no_grad():
        for i in range(min(args.num, len(ds))):
            img, _, name = ds[i]
            img = img.unsqueeze(0).to(device)
            logits = model(img)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu()
            vis = overlay(img[0], pred)
            plt.imsave(os.path.join(args.out_dir, name.replace('.nii','').replace('.gz','') + '_pred.png'), vis)

    print('Saved predictions to', args.out_dir)

if __name__ == '__main__':
    main()
