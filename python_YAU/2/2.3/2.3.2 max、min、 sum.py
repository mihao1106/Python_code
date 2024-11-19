#2.3.2 max、min、 sum

#1
x = [1,2,3,4,5]
print(max(x))
print(min(x))
print(sum(x))
print(sum(x)/len(x))
print(max(['55','111']))
print(max(['55','111']))
print(max(['999','1111'],key = len))
print(max(['abc','ABD'],key = str.upper))

#2
from random import choices
data = [choices(range(10),k = 8)]
for row in data:
    print(row)
print(max(data,key = sum))
print(max(data,key = min))
print(max(data,key = lambda row :row [7]))