# count_dataset.py
from pathlib import Path
from collections import Counter, defaultdict
import re, sys

# 修改为你的数据根目录
DATA_ROOT = Path(r"D:\COMP3710\HipMRI_Study_open")
IMAGES_DIR = DATA_ROOT / "semantic_MRs"
LABELS_DIR = DATA_ROOT / "semantic_labels_only"
SPLIT_DIR  = Path("./splits")  # 若没有可忽略

# 解析文件名：B006_Week0_LFOV.nii.gz -> pid=B006, week=0, base=B006_Week0
PAT = re.compile(r"(?P<pid>[A-Za-z]\d{3})_Week(?P<week>\d+)_LFOV\.nii\.gz$")

def scan_cases():
    cases = []
    missing_labels = []
    for img in IMAGES_DIR.glob("*_LFOV.nii.gz"):
        m = PAT.search(img.name)
        if not m:
            continue
        pid = m.group("pid")
        week = int(m.group("week"))
        base = f"{pid}_Week{week}"
        lab_name = img.name.replace("_LFOV.nii.gz", "_SEMANTIC.nii.gz")
        lab_path = LABELS_DIR / lab_name
        if not lab_path.exists():
            missing_labels.append(base)
        cases.append({"base": base, "pid": pid, "week": week, "img": img, "lab_exists": lab_path.exists()})
    return cases, missing_labels

def load_splits():
    """读取 splits/train.txt,val.txt,test.txt（若存在），返回 dict[str, set[str]]"""
    split_files = {}
    for name in ["train", "val", "test"]:
        p = SPLIT_DIR / f"{name}.txt"
        if p.exists():
            split_files[name] = {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
    return split_files if split_files else None

def summarize(cases, splits=None):
    print("=== 数据概览 ===")
    print(f"图像文件数: {len(cases)}")
    print(f"唯一病人(患者)数: {len({c['pid'] for c in cases})}")
    print(f"标签缺失数: {sum(not c['lab_exists'] for c in cases)}")

    week_cnt = Counter(c["week"] for c in cases)
    print("\n各周(Week)数量：")
    for w in sorted(week_cnt):
        print(f"  Week{w}: {week_cnt[w]}")

    if splits:
        print("\n=== Split 统计 (基于 splits/*.txt) ===")
        # 统计每个 split 的样本数
        all_bases = {c["base"] for c in cases}
        for name, items in splits.items():
            unknown = items - all_bases
            if unknown:
                print(f"  [警告] {name}.txt 中有 {len(unknown)} 个不在数据集内的条目，例如: {sorted(list(unknown))[:3]}")

        for name in ["train", "val", "test"]:
            n = len(splits.get(name, set()))
            print(f"  {name:>5}: {n}")

        # 检查患者泄漏（同一个 pid 同时出现在多个 split）
        pid_in_split = defaultdict(set)  # pid -> {split names}
        base2pid = {c["base"]: c["pid"] for c in cases}
        for split_name, items in splits.items():
            for b in items:
                pid = base2pid.get(b)
                if pid:
                    pid_in_split[pid].add(split_name)
        leaked = {pid: s for pid, s in pid_in_split.items() if len(s) > 1}
        if leaked:
            print("\n[严重] 发现患者泄漏（同一 pid 出现在多个 split）:")
            for pid, where in list(leaked.items())[:10]:
                print(f"  {pid}: {sorted(where)}")
        else:
            print("\n患者泄漏检查：未发现泄漏 ✅")

def main():
    if not IMAGES_DIR.exists():
        print(f"未找到图像目录: {IMAGES_DIR}", file=sys.stderr); sys.exit(1)

    cases, missing_labels = scan_cases()
    if missing_labels:
        print(f"[提示] 有 {len(missing_labels)} 个样本缺失标签，例如: {missing_labels[:5]}")
    splits = load_splits()
    summarize(cases, splits)

if __name__ == "__main__":
    main()
