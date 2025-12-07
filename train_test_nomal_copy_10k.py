import os
import random
import shutil
from glob import glob
from pathlib import Path

# ====== 경로 ======
ROOT = Path("./dataset_original")  # 너의 구조 그대로

SRC_TRAIN_NORMAL = ROOT / "train/normal"
DST_TRAIN_10K    = ROOT / "train/normal_5k"

SRC_TEST_NORMAL  = ROOT / "test/normal"
DST_TEST_5K      = ROOT / "test/normal_3k"

# ====== 개수 설정 ======
N_TRAIN = 5000
N_TEST  = 3000

# ====== 공용 함수 ======
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

# ====== 실행 ======
copy_random(SRC_TRAIN_NORMAL, DST_TRAIN_10K, N_TRAIN)
copy_random(SRC_TEST_NORMAL,  DST_TEST_5K,     N_TEST)