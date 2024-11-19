#5.3.2集合运算
a_set = set([8,9,10,11,12,13])
b_set = {0,1,2,3,7,8}
print(a_set | b_set)
print(a_set & b_set)
print(a_set - b_set)
print(a_set ^ b_set)

print({1,2,3}<{1,2,3,4})
print({1,2,3}<={1,2,3})
print({1,2,5}>{1,2,4})