#8.1.3函数嵌套定义
from functools import reduce
def myReduce(num, c):
    if not all (map(lambda i: 0<=int(i)<c,str(num))):
        return 'Error'
    def func(x,y):
        return x*c + y
    return reduce(func,map(int,str(num)))
print(myReduce(111,2))


def myMap(iterable,op,value):
    if op not in ('+','-','*','/','//','%','**'):
        return 'Error operator'
    def func (i):
        return eval (str(i)+op+str(value))
    return map(func,iterable)
print(list(myMap(range(5),'+',5)))
print(list(myMap(range(5),'-',5)))
print(list(myMap(range(5),'*',5)))
print(list(myMap(range(5),'/',5)))
print(list(myMap(range(5),'//',5)))
print(list(myMap(range(5),'%',5)))
print(list(myMap(range(5),'**',5)))