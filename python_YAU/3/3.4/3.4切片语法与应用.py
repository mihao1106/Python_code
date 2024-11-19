#3.4切片语法与应用
aList = [3,4,5,6,7,9,11,13,15,17]
print(aList[:])
print(aList[::-1])  #逆序列
print(aList[::2])   #下标从0开始，隔一个取一个
print(aList[::3])   #隔两个取一个
print(aList[0:3])   #从开始到结束的位置，下标为3的不算
print(aList[3:6])
print(aList[3:1009])#后边没有的就截断不写

aList = [3,5,7]
aList[len(aList):] = [9]   #尾部添加元素
aList[:0] = [1,2]          #首部添加元素
aList[3:3] = [4]           #中间添加元素
print(aList)

aList = [3,5,7,9]
print(aList[:3])
aList[:3] = [1,2,3]
print(aList)
print(aList[3:])
aList[3:] = [4,5,6,7]
print(aList)
print(aList[::2])
aList[::2] = ['a','b','c','d']
print(aList)
aList[::2] = [1]
print(aList)

aList = [3,5,7,9]
aList[:3] = []
print(aList)

aList = [3,5,7,9,11]
del aList[:3]
print(aList)

aList = [3,5,7,9,11]
del aList[::2]
print(aList)

maxNumber = int (input('请输入一个自然数：'))
lst = list(range(2,maxNumber))
m = int(maxNumber**0.5)
for index,value in enumerate(lst):
    if value > m:
        break
    lst[index + 1:] = filter(lambda x: x%value != 0,lst[index + 1:])
print(lst)