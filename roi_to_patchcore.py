from pathlib import Path
import shutil
import random

ROOT = Path("./final_dataset")

SRC_TRAIN_NORMAL = ROOT / "train/normal_roi"
SRC_TEST_NORMAL  = ROOT / "test/normal_roi"

DST_TRAIN_7000   = ROOT / "train/normal_7000"
DST_TRAIN_5000   = ROOT / "train/normal_5000"   # ★ 추가
DST_TEST_3000    = ROOT / "test/normal_3000"

SRC_TEST_ABNORMAL = ROOT / "test/abnormal_roi"
DST_TEST_PATCH_ABNORMAL = ROOT / "test/abnormal_patch"

NUM_TRAIN_NORMAL = 7000
NUM_TRAIN_SUB    = 5000       # ★ 7000 중에서 다시 뽑을 개수
NUM_TEST_NORMAL  = 3000
NUM_TEST_ABNORMAL = 3000


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_files(files, dst: Path):
    ensure_dir(dst)
    for f in files:
        shutil.copy(f, dst / f.name)


def split_normal_no_duplicate(src_train: Path, src_test: Path, num_train: int, num_test: int):
    # 두 폴더의 파일을 모두 합쳐서 풀 리스트 만들기
    files = []
    for folder in [src_train, src_test]:
        files.extend([p for p in folder.iterdir()
                      if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    total = len(files)

    if total < (num_train + num_test):
        print(f"[WARN] 정상 데이터 총합 {total}개, 요청한 {num_train+num_test}개보다 적음 → 가능한 만큼만 사용")

    # 중복 없는 랜덤 분할
    random.shuffle(files)

    train_files = files[:num_train]
    test_files  = files[num_train:num_train+num_test]

    print(f"\n=== 정상 데이터 분할 완료 ===")
    print(f"Train 정상: {len(train_files)}개")
    print(f"Test 정상:  {len(test_files)}개\n")

    return train_files, test_files


def main():
    # 1. 정상 데이터 7000 / 3000 분할 (중복 없음)
    train_normal_files, test_normal_files = split_normal_no_duplicate(
        SRC_TRAIN_NORMAL,
        SRC_TEST_NORMAL,
        NUM_TRAIN_NORMAL,
        NUM_TEST_NORMAL,
    )

    # train 7000 복사
    copy_files(train_normal_files, DST_TRAIN_7000)

    # 7000 중에서 5000개만 골라서 normal_5000에 추가 복사
    if NUM_TRAIN_SUB > len(train_normal_files):
        raise ValueError("NUM_TRAIN_SUB이 train_normal_files 개수보다 큼")

    # train_normal_files 자체가 이미 random.shuffle 된 상태라
    # 앞에서 5000개 자르는 것만으로도 랜덤 서브셋이 된다.
    sub_train_files = train_normal_files[:NUM_TRAIN_SUB]
    copy_files(sub_train_files, DST_TRAIN_5000)

    # test 3000 복사
    copy_files(test_normal_files, DST_TEST_3000)

    # 2. 비정상 test는 필요하면 여기서 다시 활성화
    # abnormal_files = [p for p in SRC_TEST_ABNORMAL.iterdir()
    #                   if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    # random.shuffle(abnormal_files)
    # selected_abnormal = abnormal_files[:NUM_TEST_ABNORMAL]
    # copy_files(selected_abnormal, DST_TEST_PATCH_ABNORMAL)

    print("\n🎉 모든 샘플링 및 복사 완료!\n")


if __name__ == "__main__":
    main()
