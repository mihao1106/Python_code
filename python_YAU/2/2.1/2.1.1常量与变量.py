#2.1.1常量与变量
x = 3
print(x)
print(type(x))
y = 5.3
print(type(y))
z = 'Hello World'
print(type(z))
w = [1,2,3]
print(type(w))
h = (10,20,30)
print(type(h))

x = 3
y = x
print(id(x))
print(id(y))
x += 6
print(x)
print(y)
print(id(x))
print(id(y))
