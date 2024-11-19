#2.3.7 filter    #过滤筛选函数
sep = ['123','hello',',.!','567abc']
print(list(filter(str.isdigit,sep)))

data = list(range(20))
print(data)

filterObject = filter(lambda x:x % 2 == 1,data)
print(filterObject)
print(3 in filterObject)
print(list(filterObject))
print(list(filterObject))
print(list(filter(None,range(-3,5))))