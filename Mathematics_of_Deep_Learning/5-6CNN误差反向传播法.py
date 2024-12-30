'''
实现了一个前向传播过程和一个基础的神经网络结构，用于处理手写数字识别问题。
代码中包含了输入层、特征映射层（可以理解为卷积层），以及输出层。
'''

import numpy as np

# 第1步：读入数据
x_data = np.array(
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
     0, ]).reshape([6, 6])
print("输入数据 x_data:")
print(x_data)

# 真实标签
target_digit_1 = 1
target_digit_2 = 0
target_digit_3 = 0

# 学习率
alpha = 0.2

# 特征映射层权重和偏置
w_feature_map_1 = np.array([-1.277, -0.454, 0.358, 1.138, -2.398, -1.664, -0.794, 0.899, 0.675]).reshape([3, 3])
w_feature_map_2 = np.array([-1.274, 2.338, 2.301, 0.649, -0.339, -2.054, -1.022, -1.204, -1.900]).reshape([3, 3])
w_feature_map_3 = np.array([-1.869, 2.044, -1.290, -1.710, -2.091, -2.946, 0.201, -1.323, 0.207]).reshape([3, 3])
b_feature_map_1 = -3.363
b_feature_map_2 = -3.176
b_feature_map_3 = -1.739

# 输出层权重和偏置
w_output_1_p1 = np.array([-0.276, 0.124, -0.961, 0.718]).reshape([2, 2])
w_output_1_p2 = np.array([-3.680, -0.594, 0.280, -0.782]).reshape([2, 2])
w_output_1_p3 = np.array([-1.475, -2.010, -1.085, -0.188]).reshape([2, 2])
w_output_2_p1 = np.array([0.010, 0.661, -1.591, 2.189]).reshape([2, 2])
w_output_2_p2 = np.array([1.728, 0.003, -0.250, 1.898]).reshape([2, 2])
w_output_2_p3 = np.array([0.238, 1.589, 2.246, -0.093]).reshape([2, 2])
w_output_3_p1 = np.array([-1.322, -0.218, 3.527, 0.061]).reshape([2, 2])
w_output_3_p2 = np.array([0.613, 0.218, -2.130, -1.678]).reshape([2, 2])
w_output_3_p3 = np.array([1.236, -0.486, -0.144, -1.235]).reshape([2, 2])
b_output_1 = 2.060
b_output_2 = -2.746
b_output_3 = -1.818

# 打印权重和偏置
print("特征映射1权重 w_feature_map_1:")
print(w_feature_map_1)
print("特征映射2权重 w_feature_map_2:")
print(w_feature_map_2)
print("特征映射3权重 w_feature_map_3:")
print(w_feature_map_3)
print("特征映射1偏置 b_feature_map_1:")
print(b_feature_map_1)
print("特征映射2偏置 b_feature_map_2:")
print(b_feature_map_2)
print("特征映射3偏置 b_feature_map_3:")
print(b_feature_map_3)

# 计算特征映射
z_feature_map_1 = np.array([(x_data[0:3, 0:3] * w_feature_map_1).sum(), (x_data[0:3, 1:4] * w_feature_map_1).sum(),
                            (x_data[0:3, 2:5] * w_feature_map_1).sum(), (x_data[0:3, 3:6] * w_feature_map_1).sum(),
                            (x_data[1:4, 0:3] * w_feature_map_1).sum(), (x_data[1:4, 1:4] * w_feature_map_1).sum(),
                            (x_data[1:4, 2:5] * w_feature_map_1).sum(), (x_data[1:4, 3:6] * w_feature_map_1).sum(),
                            (x_data[2:5, 0:3] * w_feature_map_1).sum(), (x_data[2:5, 1:4] * w_feature_map_1).sum(),
                            (x_data[2:5, 2:5] * w_feature_map_1).sum(), (x_data[2:5, 3:6] * w_feature_map_1).sum(),
                            (x_data[3:6, 0:3] * w_feature_map_1).sum(), (x_data[3:6, 1:4] * w_feature_map_1).sum(),
                            (x_data[3:6, 2:5] * w_feature_map_1).sum(),
                            (x_data[3:6, 3:6] * w_feature_map_1).sum()]).reshape(4, 4) + b_feature_map_1
print("特征映射1加权输入和偏置 z_feature_map_1:")
print(z_feature_map_1)


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 计算特征映射层激活值
a_feature_map_1 = sigmoid(z_feature_map_1)
a_feature_map_2 = sigmoid(z_feature_map_2)
a_feature_map_3 = sigmoid(z_feature_map_3)

# 计算特征映射层最大池化值
max_pooling_1 = np.array([a_feature_map_1[0:2, 0:2].max(), a_feature_map_1[0:2, 2:].max(),
                          a_feature_map_1[2:, 0:2].max(), a_feature_map_1[2:, 2:].max()]).reshape([2, 2])
max_pooling_2 = np.array([a_feature_map_2[0:2, 0:2].max(), a_feature_map_2[0:2, 2:].max(),
                          a_feature_map_2[2:, 0:2].max(), a_feature_map_2[2:, 2:].max()]).reshape([2, 2])
max_pooling_3 = np.array([a_feature_map_3[0:2, 0:2].max(), a_feature_map_3[0:2, 2:].max(),
                          a_feature_map_3[2:, 0:2].max(), a_feature_map_3[2:, 2:].max()]).reshape([2, 2])

# 计算输出层
z_output_layer_1 = (max_pooling_1 * w_output_1_p1).sum() + (max_pooling_2 * w_output_1_p2).sum() + (
            max_pooling_3 * w_output_1_p3).sum() + b_output_1
z_output_layer_2 = (max_pooling_1 * w_output_2_p1).sum() + (max_pooling_2 * w_output_2_p2).sum() + (
            max_pooling_3 * w_output_2_p3).sum() + b_output_2
z_output_layer_3 = (max_pooling_1 * w_output_3_p1).sum() + (max_pooling_2 * w_output_3_p2).sum() + (
            max_pooling_3 * w_output_3_p3).sum() + b_output_3

a_output_layer_1 = sigmoid(z_output_layer_1)
a_output_layer_2 = sigmoid(z_output_layer_2)
a_output_layer_3 = sigmoid(z_output_layer_3)

# 计算代价函数
cost = 1 / 2 * ((target_digit_1 - a_output_layer_1) ** 2 + (target_digit_2 - a_output_layer_2) ** 2 + (
            target_digit_3 - a_output_layer_3) ** 2)
print("代价函数值 Cost:")
print(cost)

# 第4步：计算delta
# 输出层delta
delta_output_layer_1 = (a_output_layer_1 - target_digit_1) * (sigmoid(z_output_layer_1)) * (
            1 - sigmoid(z_output_layer_1))
delta_output_layer_2 = (a_output_layer_2 - target_digit_2) * (sigmoid(z_output_layer_2)) * (
            1 - sigmoid(z_output_layer_2))
delta_output_layer_3 = (a_output_layer_3 - target_digit_3) * (sigmoid(z_output_layer_3)) * (
            1 - sigmoid(z_output_layer_3))
# 打印输出层误差
print("输出层误差 delta_output_layer_1:", delta_output_layer_1)
print("输出层误差 delta_output_layer_2:", delta_output_layer_2)
print("输出层误差 delta_output_layer_3:", delta_output_layer_3)

# 特征映射层delta
delta_feature_map_1_11 = (delta_output_layer_1 * w_output_1_p1[0, 0]) * sigmoid(z_feature_map_1[0, 0]) * (
            1 - sigmoid(z_feature_map_1[0, 0]))
delta_feature_map_1_12 = (delta_output_layer_1 * w_output_1_p1[0, 1]) * sigmoid(z_feature_map_1[0, 1]) * (
            1 - sigmoid(z_feature_map_1[0, 1]))
# ...（此处省略了其他delta的计算，需要按照实际情况继续计算）

# 打印特征映射层误差
print("特征映射1层误差 delta_feature_map_1_11:", delta_feature_map_1_11)
print("特征映射1层误差 delta_feature_map_1_12:", delta_feature_map_1_12)
# ...（继续打印其他delta值）

# 第5步：根据神经单元误差（delta）计算代价函数C的偏导数
# 对偏置的偏导数
db_output_1 = delta_output_layer_1
db_output_2 = delta_output_layer_2
db_output_3 = delta_output_layer_3
# ...（此处省略了其他偏置的偏导数计算）

# 对权重的偏导数
dw_output_1_p1 = delta_output_layer_1 * max_pooling_1[0, 0]
dw_output_1_p2 = delta_output_layer_1 * max_pooling_1[0, 1]
# ...（此处省略了其他权重的偏导数计算）

# 更新权重和偏置
w_output_1_p1 += alpha * dw_output_1_p1
w_output_1_p2 += alpha * dw_output_1_p2
# ...（此处省略了其他权重和偏置的更新）

# 打印更新后的权重和偏置
print("更新后的输出层权重 w_output_1_p1:")
print(w_output_1_p1)
print("更新后的输出层权重 w_output_1_p2:")
print(w_output_1_p2)
# ...（继续打印其他更新后的权重和偏置）
