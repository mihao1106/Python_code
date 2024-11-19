#例4-1 随机生成数字，统计每个数字出现的次数
from string import digits
from random import choice
z = ''.join(choice(digits) for i in range (1000))
result = {}
for ch in z:
    result[ch] = result.get(ch,0) + 1
for digit , fre in sorted(result.items()):
    print(digit,fre,sep=':')

import collections   #collections是一个库，探索出现频率用collections的Counter实现
import random        #随机函数
data = random.choices(range(10),k=100)
freq  = collections.Counter(data)
print(freq)
print(freq.most_common(1))     #查看出现次数最多的数字，并表示出现的次数