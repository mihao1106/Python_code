#5.3.3内置函数对集合的操作
x = {1,8,30,2,5}
print(x)
print(len(x))
print(max(x))
print(sum(x))
print(sorted(x))
print(list(map(str,x)))
print(list(filter(lambda item:item%5==0,x)))
print(list(enumerate(x)))
print(all(x))
print(any(x))
print(list(zip(x)))
print(list(reversed(x)))