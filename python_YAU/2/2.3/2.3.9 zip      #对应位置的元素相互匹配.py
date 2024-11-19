# 2.3.9 zip      #对应位置的元素相互匹配
print(list(zip('abcdef', '1237')))
print(tuple(zip('abcdef', '1237')))
print(list(zip('abcdef', '1237', '!@#$')))

for item in zip('abcd', range(3)):
    print(item)

x = zip('abcd', '1234')
print(list(x))
print(list(x))  # zip函数只能遍历一次
