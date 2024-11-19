# 例8-11 有奖项目节目
from random import randrange


def init():
    result = {i: 'goat' for i in range(3)}
    r = randrange(3)
    result[r] = 'car'
    return result


def startGame():
    doors = init()
    while True:
        try:
            firstDoorNum = int(input('Choose a door to open:'))
            assert 0 <= firstDoorNum <= 2
            break
        except:
            print('Door number must be between {} and {}'.format(0, 2))
    for door in doors.keys() - {firstDoorNum}:
        if doors[door] == 'goat':
            print('"goat" behind the door', door)
            thirdDoor = (doors.keys() - {door, firstDoorNum}).pop()
            change = input('Switch to {}?(y/n)'.format(thirdDoor))
            finalDoorNum = thirdDoor if change == 'y' else firstDoorNum
            if doors[finalDoorNum] == 'goat':
                return 'I Win!'
            else:
                return 'You Win.'


while True:
    print('=' * 30)
    print(startGame())
    r = input('Do you want to try once more?(y/n)')
    if r == 'n':
        break
