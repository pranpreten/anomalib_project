import pandas as pd
import os
import shutil

DATA_DIR = "./archive"
CSV_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")

print("=== 메타데이터 로딩 ===")
df = pd.read_csv(CSV_PATH)

df_normal = df[df["Finding Labels"] == "No Finding"].copy()
df_abnormal = df[df["Finding Labels"] != "No Finding"].copy()

normal_images = df_normal["Image Index"].tolist()
abnormal_images = df_abnormal["Image Index"].tolist()

print("정상:", len(normal_images))
print("비정상:", len(abnormal_images))

# 1. DATA_DIR 전체를 뒤져서 파일 이름 -> 경로 매핑 만들기
print("\n=== 이미지 인덱싱 시작 ===")
file_index = {}  # 예: {"00000001_000.png": "archive/어쩌구/.../00000001_000.png"}

for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        # 필요하면 확장자 필터링 (png/jpg 섞여 있을 수도 있으니까)
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            if fname in file_index:
                # 같은 이름 파일이 여러 폴더에 있을 경우 경고만 찍고 첫 번째 것 사용
                print(f"[WARN] 중복 파일명 발견: {fname} -> {file_index[fname]} / {os.path.join(root, fname)}")
            else:
                file_index[fname] = os.path.join(root, fname)

print(f"인덱싱된 이미지 수: {len(file_index)}")

# 2. 출력 경로
train_dir = "./dataset/train/normal"
test_normal_dir = "./dataset/test/normal"
test_abnormal_dir = "./dataset/test/abnormal"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_normal_dir, exist_ok=True)
os.makedirs(test_abnormal_dir, exist_ok=True)

# 3. 정상 → train + test_normal (80:20 split)
split_idx = int(len(normal_images) * 0.8)
train_imgs = normal_images[:split_idx]
test_norm_imgs = normal_images[split_idx:]

# 4. 복사 함수
def copy_images(img_list, target_dir):
    copied = 0
    for img_name in img_list:
        img_name = img_name.strip()

        if img_name in file_index:
            src = file_index[img_name]
            shutil.copy(src, target_dir)
            copied += 1
        else:
            print(f"[WARN] 인덱스에 없는 파일: {img_name}")

    print(f"{target_dir} → {copied}개 복사 완료")

print("\n=== 정상 이미지 복사 (train) ===")
copy_images(train_imgs, train_dir)

print("\n=== 정상 이미지 복사 (test/normal) ===")
copy_images(test_norm_imgs, test_normal_dir)

print("\n=== 비정상 이미지 복사 (test/abnormal, 전체 다) ===")
copy_images(abnormal_images, test_abnormal_dir)

print("\n완료!")
