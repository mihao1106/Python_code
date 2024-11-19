#2.3.6 reduce
def add(x,y):
    return x+y

from functools import reduce
print(reduce(add,range(1,10)))

def mul(x,y):
    return x*y
print(reduce(mul,range(1,10)))
def func(x,y):
    return x*10 + y
print(reduce(func,range(1,10)))

sets = [{1,2},{2,3},{3,4},{5,6}]
def union(x,y):
    return x|y
print(reduce(union,sets))