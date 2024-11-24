"""
输出说明

方法一：
使用 SciPy 库进行数值积分，计算了0到3的多项式函数的积分，并将结果输出为分数形式。
结果包括每个被积函数的积分结果，输出格式为“方法1本题中的积分约等于 ...”。
精度检查：
检查了所得结果与实际计算的关系，并输出是否成立的判断。

方法二：
提供了另一种增广矩阵的情况并计算了对应的系数。输出结果同样以分数形式展现。

方法三：
使用机械求积法计算积分的系数，并输出结果，格式同样为“方法3本题中的积分约等于 ...”。

"""


# 方法一：使用SciPy进行数值积分
from scipy.integrate import quad
import numpy as np
from fractions import Fraction

# 定义被积函数
def integrand0(x):
    return 1

def integrand1(x):
    return x

def integrand2(x):
    return x**2

def integrand3(x):
    return x**3

def integrand4(x):
    return x**4

# 计算定积分
result0, error0 = quad(integrand0, 0, 3)  # 计算常数函数的积分
result1, error1 = quad(integrand1, 0, 3)  # 计算 x 的积分
result2, error2 = quad(integrand2, 0, 3)  # 计算 x^2 的积分
result3, error3 = quad(integrand3, 0, 3)  # 计算 x^3 的积分
result4, error4 = quad(integrand4, 0, 3)  # 计算 x^4 的积分

# 构造线性方程组的系数矩阵
coeff_matrix = [[1, 1, 1, 1],
                 [0, 1, 2, 3],
                 [0, 1, 4, 9],
                 [0, 1, 8, 27]]

# 定义增广矩阵的结果
augmented_vector = [result0, result1, result2, result3]

# 解线性方程组
coefficients = np.linalg.solve(coeff_matrix, augmented_vector)

# 将结果转化为分数形式
coefficients_fraction = [Fraction(item).limit_denominator() for item in coefficients]
print(f'方法1本题中的积分约等于 {coefficients_fraction[0]}*f(0) + {coefficients_fraction[1]}*f(1) + {coefficients_fraction[2]}*f(2) + {coefficients_fraction[3]}*f(3)')

# 检查精度
if result4 != (coefficients_fraction[0]*0 + coefficients_fraction[1]*1 + coefficients_fraction[2]*2**4 + coefficients_fraction[3]*3**4 + 3/8*8):
    print("精度为4时，等式不成立，所以精度为3")


# 方法二：直接设定增广矩阵的结果
import numpy as np
from fractions import Fraction

# 定义增广矩阵的系数
augmented_vector_method2 = [3, 9/2, 9, 81/4]

# 解线性方程组
coefficients_method2 = np.linalg.solve(coeff_matrix, augmented_vector_method2)

# 将结果转化为分数形式
coefficients_fraction_method2 = [Fraction(item).limit_denominator() for item in coefficients_method2]
print(f'方法2本题中的积分约等于 {coefficients_fraction_method2[0]}*f(0) + {coefficients_fraction_method2[1]}*f(1) + {coefficients_fraction_method2[2]}*f(2) + {coefficients_fraction_method2[3]}*f(3)')


# 方法三：机械求积-插值型求积公式
from sympy import Rational

# 定义上限
x = 3
A = [None] * 4  # 初始化系数列表

def calculate_coefficients():
    # 计算各项系数
    A[0] = Rational((-1/6)*((1/4 * (x)**4) - (2*(x)**3) + (11/2 * (x)**2) - (6*x)))
    A[1] = Rational((1/2)*(((1/4)*(x**4))-((5/3)*(x**3))+(3*(x**2))))
    A[2] = Rational((-1/2)*(((1/4)*(x**4))-((4/3)*(x**3))+((3/2)*(x**2))))
    A[3] = Rational((1/6)*((1/4)*(x**4)-x**3+x**2))

calculate_coefficients()

# 输出结果
print(f'方法3本题中的积分约等于 {A[0]}*f(0) + {A[1]}*f(1) + {A[2]}*f(2) + {A[3]}*f(3)')
