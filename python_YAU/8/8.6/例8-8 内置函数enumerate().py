# 例8-8 内置函数enumerate()
def myEnumerate(seq):
    index = 0
    for item in seq:
        yield (index, item)
        index = index + 1


for item in myEnumerate('Hello World'):
    print(item, end=' ')
