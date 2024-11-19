#4.3字典元素访问
aDict = {'age':23,'name':'dong','sex':'male'}
print(aDict['age'])
print(aDict['weight'])

if 'weight' in aDict:
    print(aDict['weight'])
else:
    print('Error')

try:
    print(aDict['height'])
except:
    print('您输入的内容不存在，请检查')

print(aDict.get('age'))
print(aDict.get('weight','Not Exists'))

sock = {'IP':'127.2.3','port':80}
print(sock)
print(list(sock))           #字典的键
print(tuple(sock.items()))  #字典的元素对应表示

for value in sock.values(): #只表示字典的值
    print(value)
print('80'in sock)
print(80 in sock)
print(80 in sock.values())  #80存在字典的值中