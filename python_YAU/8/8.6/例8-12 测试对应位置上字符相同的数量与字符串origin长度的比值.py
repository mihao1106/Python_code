# 例8-12 测试对应位置上字符相同的数量与字符串origin长度的比值
def Rate(origin, userInput):
    right = sum(map(lambda oc, uc: oc == uc, origin, userInput))
    return round(right / len(origin), 3)


origin = 'Complex is better than complicated.'
userInput = 'Complex is BETTRR than complicated.'
print(Rate(origin, userInput))
