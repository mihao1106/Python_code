#8.5生成器函数
#例8-3
def fibo():
    a ,b = 1,1
    while True:
        yield a
        a ,b = b,a+b
seq = fibo()
for num in seq:
    if num > 500:
        break
    print(num,end=' ')