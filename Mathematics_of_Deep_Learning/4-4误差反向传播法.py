import torch

# 设置随机数种子，确保每次生成的随机数相同
torch.manual_seed(100)

# 输入数据：12个样本，每个样本有1个特征
X = torch.tensor([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]).reshape(12, 1).float()

# 隐藏层参数
N_HIDDEN = 3  # 隐藏层神经元数量
N_INPUT = 1   # 输入层特征数量，这里应该是1，因为每个样本有1个特征
N_CLASSES = 2  # 输出层类别数量

# 初始化隐藏层权重和偏置
W2 = torch.randn(N_HIDDEN, N_INPUT)  # 隐藏层权重，正确的形状是 (3, 1)
b2 = torch.randn(N_HIDDEN)           # 隐藏层偏置，形状是 (3)

# 前向传播：隐藏层
Z2 = torch.matmul(X, W2.t()) + b2   # 计算隐藏层的加权输入和偏置，注意这里使用了W2的转置
A2 = torch.sigmoid(Z2)             # 应用sigmoid激活函数

# 初始化输出层权重和偏置
W3 = torch.randn(N_CLASSES, N_HIDDEN)  # 输出层权重
b3 = torch.randn(N_CLASSES)            # 输出层偏置

# 前向传播：输出层
Z3 = torch.matmul(A2, W3.t()) + b3    # 计算输出层的加权输入和偏置，注意这里使用了W3的转置
A3 = torch.sigmoid(Z3)                # 应用sigmoid激活函数

# 计算代价函数（均方误差）
# 假设目标输出为 [1, 0]，对应于两个类别
COST = 0.5 * ((1 - A3[:, 0])**2 + (0 - A3[:, 1])**2).sum()

# 打印结果
print("隐藏层激活值：\n", A2)
print("输出层激活值：\n", A3)
print("代价：", COST)