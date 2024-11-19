# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 读取数据（请替换为你的数据路径）
data = pd.read_csv('/mnt/data/training_set_2ap_loc0_nav82.csv')

# 展示数据基本信息
print(data.info())
print(data.head())

# 选择相关的特征
features = ['rssi', 'pd', 'ed', 'nav', 'interference', 'protocol']
# 假设数据集中有这些特征
target_mcs = 'mcs'  # MCS作为目标量
target_nss = 'nss'  # NSS作为目标变量

# 去除空值
data = data[features + [target_nss, target_nss]].dropna()

# 进行特征标准化
scaler = StandardScaler()
X = data[features]
y_mcs = data[target_mcs]
y_nss = data[target_nss]
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train_mcs, X_test_mcs, y_train_mcs, y_test_mcs = train_test_split(X_scaled, y_mcs, test_size=0.3, random_state=42)
X_train_nss, X_test_nss, y_train_nss, y_test_nss = train_test_split(X_scaled, y_nss, test_size=0.3, random_state=42)

# 使用随机森林回归模型预测MCS
model_mcs = RandomForestRegressor(n_estimators=100, random_state=42)
model_mcs.fit(X_train_mcs, y_train_mcs)

# 预测MCS
y_pred_mcs_train = model_mcs.predict(X_train_mcs)
y_pred_mcs_test = model_mcs.predict(X_test_mcs)

# 使用随机森林回归模型预测NSS
model_nss = RandomForestRegressor(n_estimators=100, random_state=42)
model_nss.fit(X_train_nss, y_train_nss)

# 预测NSS
y_pred_nss_train = model_nss.predict(X_train_nss)
y_pred_nss_test = model_nss.predict(X_test_nss)

# 评估MCS模型
train_mse_mcs = mean_squared_error(y_train_mcs, y_pred_mcs_train)
test_mse_mcs = mean_squared_error(y_test_mcs, y_pred_mcs_test)
train_r2_mcs = r2_score(y_train_mcs, y_pred_mcs_train)
test_r2_mcs = r2_score(y_test_mcs, y_pred_mcs_test)

print(f"MCS模型-训练集均方误差: {train_mse_mcs:.4f}, R2: {train_r2_mcs:.4f}")
print(f"MCS模型-测试集均方误差: {test_mse_mcs:.4f}, R2: {test_r2_mcs:.4f}")

# 评估NSS模型
train_mse_nss = mean_squared_error(y_train_nss, y_pred_nss_train)
test_mse_nss = mean_squared_error(y_test_nss, y_pred_nss_test)
train_r2_nss = r2_score(y_train_nss, y_pred_nss_train)
test_r2_nss = r2_score(y_test_nss, y_pred_nss_test)

print(f"NSS模型-训练集均方误差: {train_mse_nss:.4f}, R2: {train_r2_nss:.4f}")
print(f"NSS模型-测试集均方误差: {test_mse_nss:.4f}, R2: {test_r2_nss:.4f}")

# MCS预测可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test_mcs, y_pred_mcs_test, alpha=0.6, color='g')
plt.plot([y_test_mcs.min(), y_test_mcs.max()], [y_test_mcs.min(), y_test_mcs.max()], 'r--', lw=3)
plt.xlabel('真实MCS值')
plt.ylabel('预测MCS值')
plt.title('MCS真实值与预测值对比')
plt.grid(True)
plt.show()

# NSS预测可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test_nss, y_pred_nss_test, alpha=0.6, color='b')
plt.plot([y_test_nss.min(), y_test_nss.max()], [y_test_nss.min(), y_test_nss.max()], 'r--', lw=3)
plt.xlabel('真实NSS值')
plt.ylabel('预测NSS值')
plt.title('NSS真实值与预测值对比')
plt.grid(True)
plt.show()

# 可视化MCS的特征重要性
importances_mcs = model_mcs.feature_importances_
indices_mcs = np.argsort(importances_mcs)[::-1]
feature_names = np.array(features)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_mcs[indices_mcs],y=feature_names[indices_mcs],palette="viridis")
plt.title('MCS特征重要性')
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.grid(True)
plt.show()

# 可视化NSS的特征重要性
importances_nss = model_nss.feature_importances_
indices_nss = np.argsort(importances_nss)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances_nss[indices_nss], y=feature_names[indices_nss], palette="viridis")
plt.title('NSS特征重要性')
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.grid(True)
plt.show()
