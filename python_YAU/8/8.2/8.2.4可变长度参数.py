# 8.2.4可变长度参数
def demo(a, b, c, *p):
    print(a, b, c)
    print(p)


print(demo(1, 2, 3, 4, 5, 6))
print(demo(1, 2, 3, 4, 5, 6, 7, 8))


def demo(**p):
    for item in p.items():
        print(item)


print(demo(x=1, y=2, z=3))