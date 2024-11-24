'''

输出说明

输入提示：
程序会提示用户输入需要估算的sin(x)的自变量x，单位为度，程序会将其转换为弧度进行计算。

插值计算结果：
输出使用不同插值方法计算出的sin值，分别为使用x0,x1和x1,x2的插值值，以及二次插值法的值。

误差计算：
计算并输出每种插值法的误差情况，判断误差是否在可接受范围内，并分别输出结果。

'''

import math

# 已知的拉格朗日插值点： (角度（弧度），sin值)
xn = [
    (math.pi / 6, 1 / 2),  # sin(30°)
    (math.pi / 4, 1 / (2 ** 0.5)),  # sin(45°)
    (math.pi / 3, (3 ** 0.5) / 2)  # sin(60°)
]


def lagrange_one(x, flag):
    """
    拉格朗日插值法，n=1。

    参数
    ----------
    x : float
        输入的角度（弧度）。
    flag : int
        指定使用的两个点（1: 使用 x0 和 x1，2: 使用 x1 和 x2）。

    返回
    -------
    float
        插值后的 sin 值。
    """
    if flag == 1:
        # 使用 xn[0] 和 xn[1] 进行插值
        return ((x - xn[1][0]) / (xn[0][0] - xn[1][0])) * xn[0][1] + \
               ((x - xn[0][0]) / (xn[1][0] - xn[0][0])) * xn[1][1]
    elif flag == 2:
        # 使用 xn[1] 和 xn[2] 进行插值
        return ((x - xn[2][0]) / (xn[1][0] - xn[2][0])) * xn[1][1] + \
               ((x - xn[1][0]) / (xn[2][0] - xn[1][0])) * xn[2][1]


def lagrange_two(x):
    """
    拉格朗日插值法，n=2，使用三个点进行插值。

    参数
    ----------
    x : float
        输入的角度（弧度）。

    返回
    -------
    float
        插值后的 sin 值。
    """
    # 使用 xn[0], xn[1], xn[2] 进行插值
    return ((x - xn[1][0]) * (x - xn[2][0]) /
            ((xn[0][0] - xn[1][0]) * (xn[0][0] - xn[2][0]))) * xn[0][1] + \
           ((x - xn[0][0]) * (x - xn[2][0]) /
            ((xn[1][0] - xn[0][0]) * (xn[1][0] - xn[2][0]))) * xn[1][1] + \
           ((x - xn[0][0]) * (x - xn[1][0]) /
            ((xn[2][0] - xn[0][0]) * (xn[2][0] - xn[1][0]))) * xn[2][1]


def calculate_error(x, estimated_value, flag):
    """
    计算插值的误差。

    参数
    ----------
    x : float
        输入的角度（弧度）。
    estimated_value : float
        从插值计算得到的值。
    flag : int
        指定使用的插值方法。

    返回
    -------
    None
    """
    actual_value = math.sin(x)  # 真实的 sin 值
    error = actual_value - estimated_value  # 计算误差

    # 根据使用的插值方法确定可接受的误差范围
    if flag == 1:
        error_bound = abs((x - xn[0][0]) * (x - xn[1][0])) / 2
    elif flag == 2:
        error_bound = abs((x - xn[1][0]) * (x - xn[2][0])) / (2 ** 0.5)
    elif flag == 3:
        error_bound = abs((x - xn[0][0]) * (x - xn[1][0]) * (x - xn[2][0])) / (3 * 2)

    # 判断误差是否在可接受的范围内
    if abs(error) < error_bound:
        print(f"误差在可接受范围内: {error}")
    else:
        print(f"误差超出可接受范围: {error}")


# 主程序
x_input = float(input("请输入需要估算的sin函数的自变量x（以度为单位）: ")) * (math.pi / 180)  # 转换为弧度

# 计算插值值
lagrange_value_1 = lagrange_one(x_input, 1)  # 使用第一个插值方法
lagrange_value_2 = lagrange_one(x_input, 2)  # 使用第二个插值方法
lagrange_value_3 = lagrange_two(x_input)  # 使用第二次拉格朗日插值法

# 输出结果
print(f'利用x0,x1可推导出sin({x_input * (180 / math.pi)}°)的值为: {lagrange_value_1}')
print(f'利用x1,x2可推导出sin({x_input * (180 / math.pi)}°)的值为: {lagrange_value_2}')
print(f'利用sinx的二次拉格朗日插值计算可推导出sin({x_input * (180 / math.pi)}°)的值为: {lagrange_value_3}')

# 计算误差
calculate_error(x_input, lagrange_value_1, 1)
calculate_error(x_input, lagrange_value_2, 2)
calculate_error(x_input, lagrange_value_3, 3)
