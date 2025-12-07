from pathlib import Path
import shutil
import random

# ==========================================
# 경로 설정
# ==========================================
ROOT = Path("./final_dataset")

SRC_TRAIN_NORMAL = ROOT / "train/normal_roi"
DST_TRAIN_PATCH  = ROOT / "train/normal_patch"

SRC_TEST_NORMAL  = ROOT / "test/normal_roi"
DST_TEST_PATCH_NORMAL = ROOT / "test/normal_patch"

SRC_TEST_ABNORMAL = ROOT / "test/abnormal_roi"
DST_TEST_PATCH_ABNORMAL = ROOT / "test/abnormal_patch"

NUM_TRAIN_NORMAL = 7000
NUM_TEST_NORMAL = 3000
NUM_TEST_ABNORMAL = 3000


# ==========================================
# 헬퍼 함수
# ==========================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def random_copy(src: Path, dst: Path, num_files: int):
    ensure_dir(dst)

    files = [p for p in src.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    if len(files) < num_files:
        print(f"[WARN] {src} 파일 수({len(files)})가 {num_files}보다 적음 → 가능한 만큼만 복사")
        num_files = len(files)

    selected = random.sample(files, num_files)

    print(f"\n=== {src} → {dst}")
    print(f"{num_files}개 랜덤 선택하여 복사합니다 ===")

    for f in selected:
        shutil.copy(f, dst / f.name)

    print(f"완료: {dst} 에 {num_files}개 복사됨\n")


# ==========================================
# MAIN
# ==========================================
def main():
    # Train 10k 정상 샘플
    random_copy(SRC_TRAIN_NORMAL, DST_TRAIN_PATCH, NUM_TRAIN_NORMAL)

    # Test 정상 5k 샘플
    random_copy(SRC_TEST_NORMAL, DST_TEST_PATCH_NORMAL, NUM_TEST_NORMAL)

    # Test 비정상 5k 샘플
    random_copy(SRC_TEST_ABNORMAL, DST_TEST_PATCH_ABNORMAL, NUM_TEST_ABNORMAL)

    print("\n🎉 모든 이미지 랜덤 샘플링 + 복사 완료!\n")


if __name__ == "__main__":
    main()
