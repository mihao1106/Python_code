#8.3变量作用域demo展示函数
def demo():
    global x
    x = 3
    y = 4
    print(x,y)
x = 5
demo()
print(x)

del x
print(x)
demo()
print(x)

def demo():
    x = 3
    print(x)
x = 5
print(x)
demo()
print(x)