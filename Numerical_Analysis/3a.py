"""
输出说明

内积结果:
该部分显示每个基函数与原函数的内积结果，使用 内积结果 {i} 格式输出。

系数:
显示计算得出的基函数系数，输出格式为 系数 {i}。

总误差:
最后输出 总误差，说明计算得到的逼近多项式与原函数之间的误差。

"""
from sympy import symbols, exp, integrate

# 定义符号变量
x = symbols('x')

# 定义待逼近的函数 y = e^x
f = exp(x)

# 计算基函数与原函数的内积
inner_products = [None] * 4  # 初始化内积列表
inner_products[0] = integrate(f, (x, -1, 1))  # 常数基函数的内积
inner_products[0] = round(inner_products[0], 4)

inner_products[1] = integrate(x * f, (x, -1, 1))  # x基函数的内积
inner_products[1] = round(inner_products[1], 4)

# 第二个和第三个基函数的内积计算
expr2 = (((3 * (x ** 2)) - 1) / 2) * f  # P2 (Legendre多项式的第二阶)
inner_products[2] = integrate(expr2, (x, -1, 1))
inner_products[2] = round(inner_products[2], 5)

expr3 = (((5 * (x ** 3)) - 3) / 2) * f  # P3 (Legendre多项式的第三阶)
inner_products[3] = integrate(expr3, (x, -1, 1))
inner_products[3] = round(inner_products[3], 5)

# 输出内积结果
for i in range(4):
    print(f'内积结果 {i}: {inner_products[i]}')

# 计算基函数的系数
coefficients = [None] * 4  # 初始化系数列表
for i in range(4):
    coefficients[i] = ((2 * i + 1) / 2) * inner_products[i]  # 计算系数
    coefficients[i] = round(coefficients[i], 4)

# 输出系数结果
for i in range(4):
    print(f'系数 {i}: {coefficients[i]}')

# 构建三次最佳平方逼近多项式
# 使用Legendre多项式的线性组合
approx_poly = coefficients[0] * 1 + coefficients[1] * x + coefficients[2] * expr2 + coefficients[3] * expr3

# 计算误差
wc0 = integrate(f ** 2, (x, -1, 1))  # 原函数的L2范数
wc0 = round(wc0, 6)

wc1 = 0.0  # 初始化误差贡献
wclist = [None] * 4  # 初始化误差列表
for i in range(4):
    wclist[i] = (2 / (2 * i + 1)) * (coefficients[i]) ** 2  # 计算每个基函数的误差贡献
    wc1 += wclist[i]  # 累加误差
    wc1 = round(wc1, 6)  # 四舍五入

# 计算总误差
total_error = round(wc0 - wc1, 6)

# 输出总误差
print(f'总误差: {total_error}')
