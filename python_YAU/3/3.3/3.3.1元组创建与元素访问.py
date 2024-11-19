#3.3.1元组创建与元素访问
x = (1,2,3)
print(x[0])
print(x[-1])
print(x[2])

x = (3,)
print(x)
x = (5)
print(x)
x = 3,5,7
print(x)
x = ()
print(x)
x = tuple()
print(tuple(range(5)))
print(tuple(map(str,range(5))))
print(tuple({3,4,4,5,6}))