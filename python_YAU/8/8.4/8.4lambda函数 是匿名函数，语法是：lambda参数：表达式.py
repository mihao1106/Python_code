# 8.4lambda函数 是匿名函数，语法是：lambda参数：表达式
from random import sample

data = [sample(range(100), 10) for i in range(5)]
for row in data:
    print(row)

for row in sorted(data):
    print(row)

for row in sorted(data, key=lambda row: row[1]):
    print(row)

from functools import reduce

print(reduce(lambda x, y: x * y, data[0]))
print(reduce(lambda x, y: x * y, data[1]))
print(list(map(lambda row: row[0], data)))
print(list(map(lambda row: row[data.index(row)], data)))
print(max(data, key=lambda row: row[-1]))

for row in filter(lambda row: sum(row) % 2 == 0, data):
    print(row)

print(reduce(lambda x, y: [xx + yy for xx, yy in zip(x, y)], data))
print(reduce(lambda x, y: list(map(lambda xx, yy: xx + yy, x, y)), data))
print(list(reduce(lambda x, y: map(lambda xx, yy: xx + yy, x, y), data)))