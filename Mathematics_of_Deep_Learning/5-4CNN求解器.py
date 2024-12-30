'''
实现了一个两层神经网络的前向传播和反向传播过程，用于二分类问题。
'''

import numpy as np

# 设置随机数种子，确保每次生成的随机数相同
np.random.seed(100)

# 手写数字图像数据（12个样本，每个样本有1个特征）
x_data = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]).reshape([12, 1])  # 手写数字为0
print(x_data)

# 真实标签
target_digit_0 = 1
target_digit_1 = 0

# 初始化隐藏层权重和偏置
w_hidden = np.random.randn(3, 12)  # 隐藏层权重
print(w_hidden)
b_hidden = np.random.randn(3, 1)  # 隐藏层偏置
print(b_hidden)
print()

# 初始化输出层权重和偏置
w_output = np.random.randn(2, 3)  # 输出层权重
print(w_output)
b_output = np.random.randn(2, 1)  # 输出层偏置
print(b_output)

# 学习率
learning_rate = 0.2

# 前向传播：隐藏层
z_hidden = np.dot(w_hidden, x_data) + b_hidden
print("隐藏层加权输入和偏置 z_hidden:")
print(z_hidden)
a_hidden = 1 / (1 + np.exp(-z_hidden))
print("隐藏层激活值 a_hidden:")
print(a_hidden)

# 前向传播：输出层
z_output = np.dot(w_output, a_hidden) + b_output
print("输出层加权输入和偏置 z_output:")
print(z_output)
a_output = 1 / (1 + np.exp(-z_output))
print("输出层激活值 a_output:")
print(a_output)

# 计算代价函数（均方误差）
cost = 0.5 * ((target_digit_0 - a_output[0])**2 + (target_digit_1 - a_output[1])**2)
print("代价函数值 Cost:")
print(cost)

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 计算输出层误差（delta）
delta_output_layer = (a_output[0] - target_digit_0) * sigmoid(z_output[0]) * (1 - sigmoid(z_output[0]))
delta_output_layer_1 = (a_output[1] - target_digit_1) * sigmoid(z_output[1]) * (1 - sigmoid(z_output[1]))
print("输出层误差 delta_output_layer:")
print([delta_output_layer, delta_output_layer_1])

# 计算隐藏层误差（delta）
delta_hidden_layer = np.outer((delta_output_layer * w_output[0, :] + delta_output_layer_1 * w_output[1, :]) * sigmoid(z_hidden) * (1 - sigmoid(z_hidden)), x_data.flatten())
print("隐藏层误差 delta_hidden_layer:")
print(delta_hidden_layer)

# 计算代价函数对偏置的偏导数
db_output_layer = delta_output_layer
db_output_layer_1 = delta_output_layer_1
db_hidden_layer = delta_hidden_layer.flatten()

# 计算代价函数对权重的偏导数
dw_output_layer = np.dot(delta_output_layer.reshape(1, -1), a_hidden.T)
dw_output_layer_1 = np.dot(delta_output_layer_1.reshape(1, -1), a_hidden.T)
dw_hidden_layer = np.dot(delta_hidden_layer.T, x_data.T)

# 打印最后一个权重偏导数作为示例
print("隐藏层权重偏导数 dw_hidden_layer:")
print(dw_hidden_layer)