"""

输出说明

函数定义：
f(x, y) 定义了用于计算的函数。

欧拉显式方法：
euler_explicit 函数实现了欧拉显式方法，计算从初始点到指定结束点的 y值。

初始条件和参数：
设定了初始的x和y值，步长h以及结束的x值。

结果输出：
最终打印出计算结果，以中文格式输出“欧拉显式方法计算结果为：”，并列出每对 x和y的值

"""


# 定义函数 f(x, y)
def f(x, y):
    return y - (2 * x) / y


# 欧拉显式方法
def euler_explicit(x0, y0, h, x_end):
    x = []  # 存储 x 值
    y = []  # 存储 y 值
    x.append(x0)  # 添加初始 x
    y.append(y0)  # 添加初始 y

    while x[-1] < x_end:
        x_next = x[-1] + h  # 计算下一个 x 值
        y_next = y[-1] + h * f(x[-1], y[-1])  # 计算下一个 y 值
        x.append(x_next)  # 添加到 x 列表
        y.append(y_next)  # 添加到 y 列表

    return x, y


# 初始条件和参数
x0 = 0  # 初始 x 值
y0 = 1  # 初始 y 值
h = 0.1  # 步长
x_end = 1  # 结束 x 值

# 执行欧拉显式方法
x, y = euler_explicit(x0, y0, h, x_end)

# 打印结果
print("欧拉显式方法计算结果为：")
for i in range(len(x)):
    print(f"x = {x[i]:.2f}, y = {y[i]:.4f}")

# 如需进行欧拉隐式方法，请在此处添加相关实现
