# batch_predict.py
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_root", required=True)
    ap.add_argument("--label_dir", default="semantic_labels_only")
    ap.add_argument("--pattern", default="*_LFOV.nii*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_if_done", action="store_true")
    args = ap.parse_args()

    # 这里：predict.py 在上一级目录
    script_dir = Path(__file__).resolve().parent              # .../Project7_3DUNet/result_summary
    predict_path = script_dir.parent / "predict.py"           # .../Project7_3DUNet/predict.py
    if not predict_path.exists():
        raise FileNotFoundError(f"predict.py not found at: {predict_path}")

    imgs = sorted(Path(args.images_dir).rglob(args.pattern))
    if args.limit and len(imgs) > args.limit:
        imgs = imgs[:args.limit]
    print(f"Will run {len(imgs)} cases -> {Path(args.out_dir).resolve()}")

    for i, img in enumerate(imgs, 1):
        stem = img.name.replace(".nii.gz","").replace(".nii","")
        print(f"[{i}/{len(imgs)}] {img.name}")

        if args.skip_if_done:
            if (Path(args.out_dir) / f"{stem}_pred.nii.gz").exists():
                print("  - skip (already exists)")
                continue

        cmd = [
            sys.executable, str(predict_path),
            "--image", str(img),
            "--weights", args.weights,
            "--out_dir", args.out_dir,
            "--label_root", args.label_root,
            "--label_dir", args.label_dir,
            "--binary_prostate", "--postprocess_lcc", "--save_nii", "--prob", "--metrics_csv",
            "--softmax_thr", "0.2", "--prob_class", "1",
            "--sw_enable", "--sw_patch", "96,128,128", "--sw_stride", "48,64,64",
            "--tta", "--grid", "--slices", "42,64,85",
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
