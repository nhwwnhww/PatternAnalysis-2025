import os, csv, json, argparse, glob
from statistics import mean, pstdev

def normpath(p):
    return os.path.normpath(os.path.abspath(p))

def read_metrics_csv(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            dice = None
            if 'dice_binary_prostate' in r and r['dice_binary_prostate']:
                dice = float(r['dice_binary_prostate'])
            elif 'dice_prostate' in r and r['dice_prostate']:
                dice = float(r['dice_prostate'])
            elif 'dice_macro' in r and r['dice_macro']:
                dice = float(r['dice_macro'])
            if dice is None:
                continue
            rows.append({
                'image': r.get('image', ''),
                'mode': r.get('mode', ''),
                'dice': dice,
                '_src': normpath(csv_path),
                '_from': 'csv'
            })
    return rows

def read_meta_json(meta_path):
    try:
        with open(meta_path, 'r') as f:
            m = json.load(f)
    except Exception:
        return None
    # 从 meta 兜底取 dice
    dice = None
    if 'dice_binary_prostate' in m and isinstance(m['dice_binary_prostate'], (int,float)):
        dice = float(m['dice_binary_prostate'])
    elif 'dice_prostate' in m and isinstance(m['dice_prostate'], (int,float)):
        dice = float(m['dice_prostate'])
    elif 'dice_macro' in m and isinstance(m['dice_macro'], (int,float)):
        dice = float(m['dice_macro'])
    if dice is None:
        return None
    image = m.get('image', '')
    mode  = 'binary' if m.get('binary_mode', False) else 'multiclass'
    return {
        'image': image,
        'mode': mode,
        'dice': dice,
        '_src': normpath(meta_path),
        '_from': 'meta'
    }

def smart_basename(impath):
    # 统一病例名显示：去掉扩展名与路径，只留文件名
    b = os.path.basename(impath) if impath else ''
    return b

def merge_under(root):
    root = normpath(root)
    # A) 递归找所有 metrics.csv（包括根目录）
    csv_paths = glob.glob(os.path.join(root, '**', 'metrics.csv'), recursive=True)
    if os.path.exists(os.path.join(root, 'metrics.csv')):
        csv_paths.append(os.path.join(root, 'metrics.csv'))

    # B) 递归找所有 *_meta.json（包括根目录）
    meta_paths = glob.glob(os.path.join(root, '**', '*_meta.json'), recursive=True)

    records = []
    for cp in csv_paths:
        records.extend(read_metrics_csv(cp))

    # 如果没在 csv 里出现过的病例，尝试用 meta 兜底
    seen = set((smart_basename(r['image']), r['mode']) for r in records)
    for mp in meta_paths:
        r = read_meta_json(mp)
        if not r:
            continue
        key = (smart_basename(r['image']), r['mode'])
        if key not in seen:
            records.append(r)
            seen.add(key)

    return records, meta_paths

def summarize_thresholds(meta_paths):
    agg = {}
    cnt = {}
    for mp in meta_paths:
        try:
            with open(mp, 'r') as f:
                m = json.load(f)
        except Exception:
            continue
        sweep = m.get('dice_threshold_sweep')
        if not sweep:
            continue
        for k, v in sweep.items():
            try:
                thr = float(k.split('@')[-1])
            except Exception:
                continue
            agg[thr] = agg.get(thr, 0.0) + float(v)
            cnt[thr] = cnt.get(thr, 0) + 1
    if not agg:
        return {}, None
    mean_by_thr = {t: agg[t] / cnt[t] for t in agg}
    best_thr = max(mean_by_thr, key=lambda t: mean_by_thr[t])
    return dict(sorted(mean_by_thr.items())), best_thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./pred3d_vis_bin_thr02', help='output root with per-case results or flat files')
    ap.add_argument('--out',  default='summary_all_metrics.csv', help='merged CSV path')
    args = ap.parse_args()

    recs, metas = merge_under(args.root)
    if not recs:
        print(f'[WARN] No metrics or meta with dice found under: {args.root}')
        return

    # 标准化 image 字段为“文件名”，防止不同目录重复
    for r in recs:
        r['image'] = smart_basename(r['image'])

    # 去重：若同一个 image+mode 在 CSV 与 META 同时出现，优先 CSV
    uniq = {}
    for r in recs:
        key = (r['image'], r['mode'])
        if key not in uniq or r['_from'] == 'csv':
            uniq[key] = r
    recs = list(uniq.values())

    # 写合并表
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image','mode','dice','source'])
        for r in sorted(recs, key=lambda x: x['image']):
            w.writerow([r['image'], r['mode'], f'{r["dice"]:.6f}', r['_src']])

    dices = [r['dice'] for r in recs]
    mu, sd = mean(dices), (pstdev(dices) if len(dices) > 1 else 0.0)
    print(f'[Summary] cases={len(dices)}  mean={mu:.4f}  std={sd:.4f}')
    print(f'[Saved] {args.out}')

    thr_stats, best_thr = summarize_thresholds(metas)
    if thr_stats:
        print('[Threshold sweep] mean Dice by thr:')
        for t, m in thr_stats.items():
            print(f'  thr={t:.2f}  mean_dice={m:.4f}')
        print(f'[Recommended global thr] {best_thr:.2f}')
    else:
        print('[Threshold sweep] No sweep info found. '
              'Enable it by running predict_debug.py with --debug --debug_thresholds.')

if __name__ == '__main__':
    main()
