#3.1.1列表的创建与删除
a_list = ['1','2','3','4']
print(a_list)

a = (1,2,3,4,5)
print(list(a))
b = range(0,10,2)
print(list(b))
c = map(str,range(10))
print(list(c))
d = zip('abcdef','1234')
print(list(d))
e = enumerate('Python')
print(list(e))
f = filter(str.isdigit, 'a1b2c3d456')
print(list(f))
x = 'Hello World'
print(list(x))
y = {1,2,3,4,5}
print(list(y))