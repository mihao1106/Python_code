import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
training_data = pd.read_csv('C:/Users/mihao/Desktop/数学建模/B题/training_set_2ap_loc0_nav82.csv')

# 选择特征
features = ['pkt_len', 'pd', 'ed', 'nav', 'sta_from_sta_0_rssi', 'sta_from_sta_1_rssi',
            'per', 'ppdu_dur', 'other_air_time', 'throughput']

# 填充缺失值
training_data['sta_from_sta_0_rssi'].fillna(training_data['sta_from_sta_0_rssi'].median(), inplace=True)
training_data['sta_from_sta_1_rssi'].fillna(training_data['sta_from_sta_1_rssi'].median(), inplace=True)

# 提取特征 (X) 和目标变量 (y)
X = training_data[features]
y = pd.to_numeric(training_data['seq_time'], errors='coerce')

# 对特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换数据为tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义LSTM模型
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 定义模型参数
input_dim = len(features)
hidden_dim = 128
output_dim = 1
num_layers = 2

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMRegressor(input_dim, hidden_dim, output_dim, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 5000
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# 预测
model.eval()
with torch.no_grad():
    y_pred_train_tensor = model(X_train_tensor.to(device))
    y_pred_test_tensor = model(X_test_tensor.to(device))

# 转换预测结果为numpy
y_pred_train = y_pred_train_tensor.cpu().numpy()
y_pred_test = y_pred_test_tensor.cpu().numpy()

# 模型评估
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# 打印训练集和测试集的均方误差和R2值
print(f"训练集均方误差: {train_mse:.4f}, R2: {train_r2:.4f}")
print(f"测试集均方误差: {test_mse:.4f}, R2: {test_r2:.4f}")

# 加载测试集1：test_set_1_2ap
test_set_2ap = pd.read_csv('C:/Users/mihao/Desktop/数学建模/B题/test_set_1_2ap.csv')

# 加载测试集2：test_set_1_3ap
test_set_3ap = pd.read_csv('C:/Users/mihao/Desktop/数学建模/B题/test_set_1_3ap.csv')

test_set_2ap.fillna(training_data.median(), inplace=True)  # 使用训练集的中位数填充
test_set_3ap.fillna(training_data.median(), inplace=True)

# 确保测试集具有与训练集相同的特征列
X_test_2ap = scaler.transform(test_set_2ap[features])
X_test_3ap = scaler.transform(test_set_3ap[features])

# 转换数据为tensor
X_test_2ap_tensor = torch.tensor(X_test_2ap, dtype=torch.float32).unsqueeze(1).to(device)
X_test_3ap_tensor = torch.tensor(X_test_3ap, dtype=torch.float32).unsqueeze(1).to(device)

# 使用训练好的模型进行预测
model.eval()
with torch.no_grad():
    y_pred_2ap_tensor = model(X_test_2ap_tensor)
    y_pred_3ap_tensor = model(X_test_3ap_tensor)

# 转换预测结果为numpy
y_pred_2ap = y_pred_2ap_tensor.cpu().numpy()
y_pred_3ap = y_pred_3ap_tensor.cpu().numpy()

# 输出预测结果
test_set_2ap['predicted_seq_time'] = y_pred_2ap
test_set_3ap['predicted_seq_time'] = y_pred_3ap

# 假设 test_set_2ap 和 test_set_3ap 包含实际的 'seq_time' 列
y_test_2ap = pd.to_numeric(test_set_2ap['seq_time'], errors='coerce')
y_test_3ap = pd.to_numeric(test_set_3ap['seq_time'], errors='coerce')

# 模型评估
ap2_mse = mean_squared_error(y_test_2ap, y_pred_2ap)
ap3_mse = mean_squared_error(y_test_3ap, y_pred_3ap)
ap2_r2 = r2_score(y_test_2ap, y_pred_2ap)
ap3_r2 = r2_score(y_test_3ap, y_pred_3ap)

# 打印测试集的均方误差和R2值
print(f"2AP测试集均方误差: {ap2_mse:.4f}, R2: {ap2_r2:.4f}")
print(f"3AP测试集均方误差: {ap3_mse:.4f}, R2: {ap3_r2:.4f}")

# 可以将预测结果导出为新的CSV文件
output_2ap_file = 'C:/Users/高高/desktop/predicted_test_set_1_2ap.csv'
output_3ap_file = 'C:/Users/高高/desktop/predicted_test_set_1_3ap.csv'

test_set_2ap.to_csv(output_2ap_file, index=False, encoding='utf-8-sig')
test_set_3ap.to_csv(output_3ap_file, index=False, encoding='utf-8-sig')

print(f"预测结果已保存至: {output_2ap_file} 和 {output_3ap_file}")

# 取出实际值和预测值
actual_2ap = test_set_2ap['seq_time']
predicted_2ap = test_set_2ap['predicted_seq_time']
actual_3ap = test_set_3ap['seq_time']
predicted_3ap = test_set_3ap['predicted_seq_time']

# 绘制 2AP 折线图
plt.figure(figsize=(14, 7))
plt.plot(actual_2ap, label='Actual 2AP', color='blue', linestyle='-')
plt.plot(predicted_2ap, label='Predicted 2AP', color='cyan', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted seq_time for 2AP')
plt.xlabel('Index')
plt.ylabel('seq_time')
plt.show()

# 绘制 3AP 折线图
plt.figure(figsize=(14, 7))
plt.plot(actual_3ap, label='Actual 3AP', color='green', linestyle='-')
plt.plot(predicted_3ap, label='Predicted 3AP', color='orange', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted seq_time for 3AP')
plt.xlabel('Index')
plt.ylabel('seq_time')
plt.show()
