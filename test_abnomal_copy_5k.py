import os
import random
import shutil
from glob import glob
from pathlib import Path

ROOT = Path("./dataset_original")

SRC_ABNORMAL = ROOT / "test/abnormal"
DST_ABNORMAL_5K = ROOT / "test/abnormal_3k"

N_ABNORMAL = 3000

def copy_random(src_dir, dst_dir, num_items):
    files = glob(str(src_dir / "*"))
    if len(files) == 0:
        print(f"[WARN] No files found in {src_dir}")
        return

    num = min(num_items, len(files))
    random.seed(42)
    selected = random.sample(files, num)

    dst_dir.mkdir(parents=True, exist_ok=True)

    for f in selected:
        shutil.copy(f, dst_dir / os.path.basename(f))

    print(f"Done: Copied {num} images → {dst_dir}")

copy_random(SRC_ABNORMAL, DST_ABNORMAL_5K, N_ABNORMAL)