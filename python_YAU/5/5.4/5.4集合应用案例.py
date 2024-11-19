#5.4集合应用案例
n = int(input('请输入一个自然数：'))
numbers = set(range(2,n))
m = int(n**0.5)+1
for p in range(2,m):
    for i in range(2,n//p+1):
        numbers.discard(i*p)
print(numbers)
