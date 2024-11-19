# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 读取数据(替换为你的文件路径)
data = pd.read_csv('C:/Users/mihao/Desktop/数学建模/B题/training_set_2ap_loc0_nav82.csv')

# 展示数据基本信息
print(data.info())
print(data.head())

# 选择相关的特征,去掉不必要的列
features = ['test_dur', 'pkt_len', 'pd', 'ed', 'nav',
            'ap_from_ap_x_sum_ant_rssi',
            'sta_from_ap_x_sum_ant_rssi', 'num_ampdu', 'mcs', 'nss', 'per']
target = 'seq_time'

# 去除空值
data = data[features + [target]].dropna()

# 进行特征标准化
scaler = StandardScaler()
X = data[features]
y = data[target]

# 将数据进行标准化处理
X_scaled = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 使用随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 模型评估
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# 打印训练集和测试集的均方误差和R2值
print(f"训练集均方误差: {train_mse:.4f}, R2:{train_r2:.4f}")
print(f"测试集均方误差: {test_mse:.4f}, R2: {test_r2:.4f}")

# 可视化真实值与预测值的对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel("真实发送时长")
plt.ylabel("预测发送时长")
plt.title('真实值与预测值对比')
plt.grid(True)
plt.show()

# 特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = np.array(features)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="coolwarm")
plt.title('特征重要性')

# 绘制特征重要性图表
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.grid(True)
plt.show()

# 信道空闲时间分析（假设到达率lambda）
lambda_rate = 0.5
t = np.linspace(0, 10, 100)
P_idle = np.exp(-lambda_rate * t)

# 可视化信道空闲概率
plt.figure(figsize=(8, 5))
plt.plot(t, P_idle, 'b-', label=f'Poisson分布$\\lambda={lambda_rate}$')
plt.fill_between(t, P_idle, alpha=0.3, color='blue')
plt.xlabel('时间')
plt.ylabel('空闲概率')
plt.title('信道空闲概率')
plt.grid(True)
plt.legend()
plt.show()