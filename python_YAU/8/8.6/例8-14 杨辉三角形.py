# 例8-14 杨辉三角形
from functools import lru_cache


@lru_cache(maxsize=64)
def cni(n, i):
    if n == i or i == 0:
        return 1
    return cni(n - 1, i) + cni(n - 1, i - 1)


def yanghui(num):
    for n in range(num):
        for i in range(n + 1):
            print(str(cni(n, i)).ljust(4), end=' ')
        print()


yanghui(8)
