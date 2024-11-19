#例3—3
xList = [1,2,3]
yList = [3,1,4]
print([(x,y) for x,y in zip(xList,yList)])#把zip对应位置上的元素配对的表示出来
print([(x,y) for x in xList for y in yList])#x和y进行组合
print([(x,y) for x in xList if x == 1 for y in yList])
print([(x,y) for x in xList if x == 1 for y in yList if y!=x])

result = []
for x in xList:
 if x==1:
     for y in yList:
         if y!=x:
             result.append((x,y))
print(result)