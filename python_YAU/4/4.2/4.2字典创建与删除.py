#4.2字典创建与删除
aDict = {'IP':'127.0.0.1','port':80}
print(aDict)

x = dict()
x = {}
keys = ['a','b','c','d']
values = [1,2,3,4]
aDict = dict(zip(keys,values))
print(aDict)

a = ['h','i','j','k']
b = [1,2,3,4]
aDict = dict(zip(a,b))
print(aDict)

aDict = dict(name='Dong',age = 39)
print(aDict)

aDict = dict.fromkeys(['name','age','sex'])
print(aDict)

del aDict
words = ['Hello','Python','World']             #用字典推导式创建字典
print({word:word.upper() for word in words})