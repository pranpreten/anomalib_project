import os
import cv2
from tqdm import tqdm

# ========================
# ROI CROP FUNCTION
# ========================
def crop_roi(img):
    """NIH X-ray 형태 기준 ROI 크롭"""

    H, W = img.shape[:2]

    # NIH X-ray 표준 비율 기반
    top = int(H * 0.10)
    bottom = int(H * 0.90)
    left = int(W * 0.15)
    right = int(W * 0.85)

    return img[top:bottom, left:right]


# ========================
# CROP ONE DIRECTORY
# ========================
def crop_folder(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n=== Cropping: {src_dir} → {dst_dir} (총 {len(files)}개) ===")

    for fname in tqdm(files):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)

        img = cv2.imread(src)
        if img is None:
            print(f"[WARN] 읽기 실패 → {src}")
            continue

        cropped = crop_roi(img)
        cv2.imwrite(dst, cropped)


# ========================
# MAIN PIPELINE
# ========================
def main():
    # 너 작업 경로 기준
    ROOT = "./dataset_original"

    # (src_rel, dst_rel)
    paths = [
        ("train/normal_5k",      "train/roi_normal_5k"),
        ("test/normal_3k",       "test/roi_normal_3k"),
        ("test/abnormal_3k",     "test/roi_abnormal_3k"),
    ]

    for src_rel, dst_rel in paths:
        src_dir = os.path.join(ROOT, src_rel)
        dst_dir = os.path.join(ROOT, dst_rel)

        if not os.path.isdir(src_dir):
            print(f"[SKIP] 소스 폴더 없음: {src_dir}")
            continue

        crop_folder(src_dir, dst_dir)

    print("\n🎉 완료! 크롭된 ROI 이미지는 dataset_original/ 안의 roi_* 폴더들에 저장됨.")


if __name__ == "__main__":
    main()