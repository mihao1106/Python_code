#3.1.3列表常用方法
#1 append、insert、extend
x = list('123')
print(x)
x.append('4')
print(x)

x = [1,2,3]
x.append(4)
print(x)            #[1, 2, 3, 4]
print(x.index(3))   #查看3在列表中的位置——2 是从0开始计数的
print(x.index(4))   #3 是从0开始计数的
'''print(x.index(8))   #8 is not in list'''
x.insert(0,0)
print(x)
print(x.index(3))
x.extend([5,6,7])
print(x)
#2 pop、remove
x = [1,2,3,4,5,6,7]
print(x.pop())
print(x.pop(0))
print(x)
x.remove(4)
print(x)
#3 count、index
x = [1,2,3,3,3,3,4,4,5,6]
print(x.count(4))
print(x.index(5))
#4 sort、reverse
x = list(range(11))
print(x)
import random
random.shuffle(x)
print(x)

x.sort(key = lambda item:len(str(item)),reverse = True)
print(x) #按照字符串的长度进行降序排列，只把两位数的提前了，其余的一位数不变
x.sort(key = str)
print(x)
x.sort()
print(x)
x.reverse()    #翻转
print(x)