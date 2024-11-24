from sympy import symbols, integrate, exp

# 定义变量
x = symbols('x')

# 定义待逼近的函数
y = exp(x)

# 计算基函数与原函数的内积
# 创建内积结果的列表
neiji = [None] * 4  # 4个基函数的内积结果

# 计算内积
neiji[0] = round(integrate(y, (x, -1, 1)), 4)  # 第一个基函数：常数1
neiji[1] = round(integrate(x * y, (x, -1, 1)), 4)  # 第二个基函数：x
expr2 = (((3 * (x)**2) - 1) / 2) * y  # 第三个基函数：Legendre多项式 P2
neiji[2] = round(integrate(expr2, (x, -1, 1)), 5)
expr3 = (((5 * (x)**3) - 3) / 2) * y  # 第四个基函数：Legendre多项式 P3
neiji[3] = round(integrate(expr3, (x, -1, 1)), 5)

# 打印内积结果
for i in range(4):
    print(f'内积结果 neiji[{i}]: {neiji[i]}')

# 计算基函数的系数
a = [None] * 4  # 存储系数的列表
for i in range(4):
    if neiji[i] is not None:  # 确保 neiji[i] 不为 None
        a[i] = round((2 * i + 1) / 2 * neiji[i], 4)
    else:
        a[i] = 0  # 如果 neiji[i] 为 None，则系数设为 0

# 打印系数结果
for i in range(4):
    print(f'系数 a[{i}]: {a[i]}')

# 计算三次最佳平方逼近多项式
# 使用基函数和计算的系数构造多项式
s = a[0] * 1 + a[1] * x + a[2] * (((3 * (x)**2) - 1) / 2) + a[3] * (((5 * (x)**3) - 3) / 2)

# 计算误差
wc0 = round(integrate(y**2, (x, -1, 1)), 6)  # 原函数的L2范数
wc1 = 0.0  # 初始化L2范数的误差
wclist = [None] * 4  # 存储每个基函数的贡献

# 计算每个基函数对误差的贡献
for i in range(4):
    if a[i] is not None:  # 确保 a[i] 不为 None
        wclist[i] = round((2 / (2 * i + 1)) * (a[i])**2, 6)
        wc1 += wclist[i]  # 累加贡献

# 计算总误差
wc = round(wc0 - wc1, 6)

# 输出最终误差
print(f'总误差 wc: {wc}')