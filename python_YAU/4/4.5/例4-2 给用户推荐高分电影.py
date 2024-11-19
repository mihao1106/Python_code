#例4-2 给用户推荐高分电影
from random import randrange
data = {'user'+str(i):{'film'+str(randrange(1,15)):randrange(1,6) for j in range(randrange(3,10))}for i in range(10)}
user = {'film' + str(randrange(1,15)):randrange(1,6)for i in range(5)}
f = lambda item:(-len(item[1].keys()&user),sum(((item[1].get(film)-user.get(film))**2 for film in user.keys()&item[1].keys())))
similarUser,films = min(data.items(),key=f)
print('know data'.center(50,'='))
for item in data.items():
    print(len(item[1].keys()&user.keys()),sum(((item[1].get(film)-user.get(film))**2 for film in user.keys()&item[1].keys())),item,sep=':')
print('current user'.center(50,'='))
print(user)
print('most similar user and his film'.center(50,'='))
print(similarUser,films,sep=':')
print('recommmended film'.center(50,'='))
print(max(films.keys()-user.keys(),key = lambda film:films[film]))