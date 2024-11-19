#例3—1
xList = list(range(5))
yList = list(range(5,10))
print(xList)
print(yList)
zList = [x + y for x,y in zip(xList,yList)]
print(zList)
hList = [x - y for x,y in zip(xList,yList)]
print(hList)
print(sum([x*y for x,y in zip(xList,yList)]))
wList = [x *5 for x in xList]
print(wList)