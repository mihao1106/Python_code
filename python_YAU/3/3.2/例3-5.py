#例3—5
x = [3,7,23,21,8,10]
avg = sum(x) / len(x)
s = [(xi-avg)**2 for xi in x ]
s = (sum(s)/len(s))** 0.5
print(s)