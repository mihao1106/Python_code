#2.3.5 map
print(list(map(str,range(5))))
print(tuple(map(str,range(5))))
x = ['aaaa','bc','d','ba']
print(x)
print(list(map(str.upper,x)))      #大写
x = ['Hello World.']
print(list(map(str.swapcase,x)))   #交换大小写
print(sum(map(int,'1234')))
print(''.join(map(lambda item:item[0],x)))

x = ['Hello','World']
print(''.join(map(lambda item:item[0],x)))


def add5(num):
    return num + 5
print(list(map(add5,range(10))))

def add(x,y):
    return x + y
print(list(map(add,range(5),range(5,10))))