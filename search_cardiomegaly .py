import os
import shutil
import pandas as pd
import cv2
from tqdm import tqdm

# ==========================
# ROI CROP 함수
# ==========================
def crop_roi(img):
    """NIH X-ray 기준으로 ROI 영역만 크롭"""
    H, W = img.shape[:2]
    top = int(H * 0.10)
    bottom = int(H * 0.90)
    left = int(W * 0.15)
    right = int(W * 0.85)
    return img[top:bottom, left:right]


# ==========================
# 설정
# ==========================
ARCHIVE_DIR = "./archive"
IMAGE_FOLDERS = [f"images_{str(i).zfill(3)}" for i in range(1, 13)]
CSV_PATH = os.path.join(ARCHIVE_DIR, "Data_Entry_2017.csv")

OUTPUT_DIR = "./final_dataset/test/abnormal_card"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# 1. CSV 읽고 Cardiomegaly만 필터링
# ==========================
df = pd.read_csv(CSV_PATH)
df_cardio = df[df["Finding Labels"].str.contains("Cardiomegaly", na=False)].copy()

print("Cardiomegaly 라벨 이미지 개수:", len(df_cardio))

cardio_images = df_cardio["Image Index"].tolist()

# ==========================
# 2. ROI 크롭 후 저장
# ==========================
for img_name in tqdm(cardio_images, desc="Cropping & Saving Cardiomegaly"):
    found = False
    for folder in IMAGE_FOLDERS:
        src_path = os.path.join(ARCHIVE_DIR, folder, "images", img_name)
        if os.path.exists(src_path):
            found = True
            
            # 이미지 로드
            img = cv2.imread(src_path)
            if img is None:
                print(f"[ERROR] 이미지 로드 실패: {src_path}")
                break

            # ROI 크롭 적용
            cropped = crop_roi(img)

            # 저장 경로
            dst_path = os.path.join(OUTPUT_DIR, img_name)

            # 저장
            cv2.imwrite(dst_path, cropped)

            break
    
    if not found:
        print(f"[WARN] 파일 없음: {img_name}")

print("\n=== 완료! Crop 적용된 Cardiomegaly 이미지 저장됨 ===")
print(f"저장 경로: {OUTPUT_DIR}")
