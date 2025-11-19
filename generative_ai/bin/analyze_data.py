import json
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ 使用 glob 读取当前目录下所有 .json 文件
file_list = glob.glob("../data/case_*/*.json")
print(f"读取到 {len(file_list)} 个 JSON 文件")

# 判断是否出现重复的模拟
for case in os.listdir("../data"):
    if os.path.isdir(os.path.join("../data", case)):
        print(glob.glob(f"../data/{case}/exodus_files/*.e")[0] + ":" + str(len(os.listdir(f"../data/{case}/npy_files"))))

# 2️⃣ 逐个加载 JSON 文件为字典并组成 DataFrame
data_list = []
for file in file_list:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data_list.append(data)

df = pd.DataFrame(data_list)

# 3️⃣ 基本统计信息
print("==== 数据统计描述 ====")
print(df.describe().T)

# 4️⃣ 相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5️⃣ 特征分布直方图
df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# 6️⃣ PCA 降维可视化（可选）
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df)
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title("PCA Feature Space Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
