#例7-6猜数游戏
from random import randint
def guess(maxValue=100,maxTimes=5):
    value = randint(1,maxValue)
    for i in range (maxTimes):
        prompt = 'Start to GUESS:' if i == 0 else 'Guess again:'
        try:
            x = int (input(prompt))
        except:
            print('Must input an integer batween 1 and ',maxValue)
        else:
            if x == value:
                print('Congratulations!')
                break
            elif x > value:
                print('Too big')
            else:
                print('Too little')
    else:
        print('Game over,FALL')
        print('The value is',value)