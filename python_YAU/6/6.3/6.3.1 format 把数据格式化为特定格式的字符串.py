#6.3.1 format 把数据格式化为特定格式的字符串
print('{0:.3f}'.format(1/3))
print('{0:%}'.format(3.5))
print('{0:.2%}'.format(3.5))
print('{0:10.2%}'.format(3.5))
print('{0:<10.2%}'.format(3.5))
print("The number {0:,} in hex is :{0:#x},in oct is {0:#o}".format(55))
print("The number {0:,} in hex is :{0:x},the number {1} in oct is {1:o}".format(5555,55))
print("The number {1} in hex is :{1:#x},the number {0} in oct is {0:#o}".format(5555,55))
print("my name is {name},my age is {age},and my QQ is {qq}".format(name = 'dong',qq = '2870871054',age = '23'))
print("I'm {age:d} years old".format(age=0o51))
print('{0:<8d},{0:^8d},{0:>8d}'.format(65))
print('{0:_},{0:_x}'.format(10000000))

name = 'dong'
age = '23'
print(f'My name is {name} , and I am {age} years old.')