#2.3.3 input()、print()
x = input('Please input:')
print(type(x))
#
password = input('Please input:')
print(f'Please input:{password}')
print(type(password))
print(type(int(password)))
print(type(eval(password)))

print(1,2,3,4,sep = '\t')
print(1,2,3,4,sep = '.')
print(1,2,3,4,end = '~')
for i in range(10):
    print(i,end = '~')