# 例8-13 测试对应位置上字符相同的数量与字符串origin长度的比值
from functools import reduce


def demo(a, n):
    a = str(a)
    return sum(map(lambda i: eval(a * i), range(1, n + 1)))


print(demo(1, 3))
print(demo(5, 4))
print(demo(9, 2))
