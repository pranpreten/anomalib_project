import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== 1. CSV 로드 ====
CSV_PATH = "./archive/Data_Entry_2017.csv"
df = pd.read_csv(CSV_PATH)

# ==== 2. 나이를 숫자로 변환 ====
df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")

# ==== 3. 정상 데이터만 ====
df_normal = df[df["Finding Labels"] == "No Finding"]

# ==== 4. 성별 컬럼 정리 ====
df_normal = df_normal[df_normal["Patient Gender"].isin(["M", "F"])]

print("=== 전체 정상 데이터 성별 분포 ===")
print(df_normal["Patient Gender"].value_counts())

# ==== 5. 성별별 나이 분포 ====
plt.figure(figsize=(12,5))
sns.histplot(
    data=df_normal,
    x="Patient Age",
    hue="Patient Gender",
    kde=True,
    bins=40,
    palette={"M":"blue", "F":"red"},
    alpha=0.5,
)

plt.xticks(np.arange(0, 101, 10))
plt.title("Age Distribution of Normal Only (Separated by Gender)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(title="Gender", labels=["Male", "Female"])
plt.show()