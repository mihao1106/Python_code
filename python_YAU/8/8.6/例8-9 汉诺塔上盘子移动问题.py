# 例8-9 汉诺塔上盘子移动问题
def hannoi(num, src, dst, temp=None):
    if num < 1:
        return
    global times
    hannoi(num - 1, src, temp, dst)
    print('The {0} Times move:{1}==>{2}'.format(times, src, dst))
    towers[dst].append(towers[src].pop())
    for tower in 'ABC':
        print(tower, ':', towers[tower])
    times += 1
    hannoi(num - 1, temp, dst, src)
    times = 1
    n = 3
    towers = {'A': list(range(n, 0, -1)),
              'B': [],
              'C': []}
    hannoi(n, 'A', 'C', 'B')
