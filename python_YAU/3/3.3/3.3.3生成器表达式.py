#3.3.3生成器表达式
g = ((i + 2)**2 for i in range (10))
print(g)
print(tuple(g))
print(list(g))
g = ((i + 2)**2 for i in range (10))
print(next(g))
print(next(g))
g = ((i + 2)**2 for i in range (10))
for item in g:
    if item > 50:
        break
    print(item,end=' ')
g = ((i + 2)**2 for i in range (10))
for item in g:
    if item > 50:
        break
    print(item,end='\t')