#例7-2买鸡问题
for x in range(21):
    for y in range(34):
        z = 100 - x - y
        if z%3==0 and 5*x + 3*y + z//3 ==100:
            print(x,y,z)