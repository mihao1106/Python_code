# 例8-10 判断一个数字是否为丑数
def demo(n):
    for i in (2, 3, 5):
        while True:
            m, r = divmod(n, i)
            if r != 0:
                break
            else:
                n = m
    return n == 1


print(demo(30))
print(demo(50))
print(demo(70))
print(demo(90))
