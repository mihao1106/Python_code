# 3.5序列解包
x, y, z = 1, 2, 3
x, y, z = (False, 3.5, 'exp')
x, y, z = [1, 2, 3]
x, y = y, x
x, y, z = range(3)
x, y, z = map(int, '123')
print(x, y, z)
s = {'a': 1, 'b': 2, 'c': 3}
b, c, d = s
print(b)
b, c, d = s.items()
print(b)
b, c, d = s.values()
print(b, c, d)

k = ['a', 'b', 'c', 'd']
v = [1, 2, 3, 4]
for k, v in zip(k, v):
    print(k, v)

x = ['a', 'b', 'c']
for i, v in enumerate(x):
    print(i, v)

s = {'a': 1, 'b': 2, 'c': 3}
for k, v in s.items():
    print(k, v)