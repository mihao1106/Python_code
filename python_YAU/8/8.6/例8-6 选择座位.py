# 例8-6 选择座位
def arrangeOrder(n):
    seats = [0] * n
    for _ in range(n):
        span = (0, 0)
        for pos in range(n):
            if seats[pos] == 0 and (pos == 0 or seats[pos - 1] == 1):
                start = pos
            elif (seats[pos] == 1 and seats[pos - 1] == 0 and
                  pos - start > span[1] - span[0]):
                span = (start, pos - 1)
        if seats[pos] == 0 and pos - start >= span[1] - span[0]:
            span = (start, pos)
        seats[(span[1] + span[0]) // 2] = 1
        print(''.join(map(str,
                          seats)).translate(''.maketrans('01', '_x')))
    arrangeOrder(18)
