#6.3.7 replace()代替取代、maketrans()、translate()
s = "Python 是一门非常优秀的编程语言。"
print(s.replace('编程','程序设计'))
pwd = '"or"a"="a'
print(s.replace('"',' '))
print(pwd.replace('"',' ').replace('=',''))

from string import ascii_lowercase as lowercase
table = ''.maketrans(lowercase,lowercase[3:]+lowercase[:3])
print(table)

text = 'Beautiful is better than ugly.'
print(text.translate(table))
table = ''.maketrans('0123456789','零一二三四五六七八九')
print('2018年12月31日'.translate(table))