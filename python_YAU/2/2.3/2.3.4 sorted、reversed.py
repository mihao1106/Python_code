#2.3.4 sorted、reversed
x = list(range(11))
import random
random.shuffle(x)
print(x)
print(sorted(x))
print(sorted(x,key = str))
print(sorted(x,key = lambda item : len(str(item)),reverse = True))

x = ['aaaa','bc','d','b','ba']
print(sorted(x,key = lambda item:(len(item),item)))

num = random.choices(range(1,10),k = 5)
print(num)

print(int(''.join(sorted(map(str,num),reverse=True))))
print(int(''.join(sorted(map(str,num)))))

data = random.choices(range(50),k = 11)
print(data)           #输出随机数列
print(sorted(data))   #对上述随机数列进行排序
print(sorted(data)[len(data)//2])

import statistics
print(statistics.median(data))

x = ['aaaa','bc','d','b','ba']
print(reversed(x))
print(list(reversed(x)))           #返回之前的对象
print(''.join(reversed('Hello World.')))    #对字符串进行翻转

y = reversed(x)
print(len(y))                   #不支持内置函数len()
print(reversed(reversed(x)))    #不支持
print('d' in y)
print('d' in y)                 #每个元素只能使用一次
print('b' in y)                 #使用过的元素不可再次使用