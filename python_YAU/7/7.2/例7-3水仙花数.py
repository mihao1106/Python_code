#例7-3水仙花数
for num in range(100,1000):
    r = map(lambda x:int(x)**3,str(num))
    if sum(r)==num:
        print(num)