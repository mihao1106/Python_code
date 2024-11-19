#例7-5验证6174猜想
from string import digits
from itertools import combinations
for item in combinations(digits,4):
    times = 0
    while True:
        big = int(''.join(sorted(item,reverse=True)))
        little = int(''.join(sorted(item)))
        difference = big-little
        times = times + 1
        if difference == 6174:
            if times>7:
                print(times)
            break
        else:
            item = str(difference)