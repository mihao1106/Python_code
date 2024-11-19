#8.2.2默认值参数
def mySum(iterable,start=0):
    for item in iterable:
        start += item
    return start
print(mySum([1,2,3,4]))
print(mySum([1,2,3,4],5))
print(mySum(['1','2','3','4'],''))
print(mySum([[1],[2],[3],[4]],[]))