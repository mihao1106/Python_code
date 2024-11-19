# 例8-5 交税
def calcFee(total, num):
    '''total为原始总收入，num为申报人数'''
    origin = 800 + (total - 800) * 0.8
    base = num * 800
    if total <= base:
        return (total, total - origin)
    now = base + (total - base) * 0.8
    return (now, now - origin)


print(calcFee(10000, 5))
print(calcFee(10000, 8))
