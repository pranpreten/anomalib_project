import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== 1. CSV 로드 ====
CSV_PATH = "./archive/Data_Entry_2017.csv"   # 너 경로 맞춰서 변경

df = pd.read_csv(CSV_PATH)

# ==== 2. 나이 숫자로 변환 ====
df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")

# ==== 3. 기본 통계량 ====
print(df["Patient Age"].describe())


# ==== 5. 정상(No Finding)만 필터 ====
df_normal = df[df["Finding Labels"]=="No Finding"]

plt.figure(figsize=(10,5))
sns.histplot(df_normal["Patient Age"], bins=40, kde=True, color="green")
plt.xticks(np.arange(0, 101, 10)) 
plt.title("Age Distribution (Normal Only)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

