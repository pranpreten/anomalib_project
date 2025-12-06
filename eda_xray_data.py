import pandas as pd
import os

DATA_DIR = "./archive"
CSV_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")

print("=== 메타데이터 로딩 ===")
df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------
# 1. 정상 / 비정상 분리
# ------------------------------------------------------
df_normal = df[df["Finding Labels"] == "No Finding"].copy()
df_abnormal = df[df["Finding Labels"] != "No Finding"].copy()

# ------------------------------------------------------
# 2. 정상 데이터 섞기 + 80/20 split
# ------------------------------------------------------
df_normal = df_normal.sample(frac=1, random_state=42)  # shuffle

split_idx = int(len(df_normal) * 0.8)
df_train_normal = df_normal.iloc[:split_idx].copy()
df_test_normal = df_normal.iloc[split_idx:].copy()

# ------------------------------------------------------
# 3. 비정상 100% test
# ------------------------------------------------------
df_test_abnormal = df_abnormal.copy()

# ------------------------------------------------------
# 4. 결과 출력 (각 DataFrame 확인)
# ------------------------------------------------------
print("\n=== df_train_normal (정상 80%) ===")
print(df_train_normal.head())
print("Shape:", df_train_normal.shape)

print("\n=== df_test_normal (정상 20%) ===")
print(df_test_normal.head())
print("Shape:", df_test_normal.shape)

print("\n=== df_test_abnormal (비정상 100%) ===")
print(df_test_abnormal.head())
print("Shape:", df_test_abnormal.shape)

# ------------------------------------------------------
# 5. Split 요약 테이블
# ------------------------------------------------------
df_overview = pd.DataFrame({
    "dataset": ["train_normal", "test_normal", "test_abnormal"],
    "count": [
        len(df_train_normal),
        len(df_test_normal),
        len(df_test_abnormal),
    ]
})

print("\n=== Split Overview ===")
print(df_overview)
