#例3—2
from random import randint
x = [randint(1,10) for i in range(20)]
print(x)
m = max(x)
print(m)
#最大整数所出现的位置
print([index for index,value in enumerate(x) if value == m])