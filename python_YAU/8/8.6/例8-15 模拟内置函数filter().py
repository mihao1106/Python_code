# 例8-15 模拟内置函数filter()
def myFilter(func, seq):
    if func is None:
        func = bool
    for item in seq:
        if func(item):
            yield item


print(list(myFilter(None, range(-3, 5))))
print(myFilter(str.isdigit, '123bcdse45'))
print(list(myFilter(lambda x: x > 5, range(10))))
