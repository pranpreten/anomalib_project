import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 1. CSV 로드 ====
CSV_PATH = "./archive/Data_Entry_2017.csv"
df = pd.read_csv(CSV_PATH)

# ==== 2. 정상 데이터만 필터 ====
df_normal = df[df["Finding Labels"] == "No Finding"]

# ==== 3. 성별(M/F)만 남기기 ====
df_normal = df_normal[df_normal["Patient Gender"].isin(["M", "F"])]

# ==== 4. 성별 개수 출력 ====
print("=== Normal Only Gender Count ===")
print(df_normal["Patient Gender"].value_counts())

# ==== 5. 성별 비율 시각화 ====
plt.figure(figsize=(6,5))
sns.countplot(data=df_normal, x="Patient Gender", palette={"M":"blue", "F":"red"})
plt.title("Gender Distribution (Normal Only)")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()