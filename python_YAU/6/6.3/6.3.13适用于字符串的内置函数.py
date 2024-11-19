#6.3.13适用于字符串的内置函数
print(ord('董'))
print(list(map(ord,'董付国')))
print(chr(33891))
text = '《玩转Python轻松过二级》'
print(max(text))
print(min(text))
print(len(text))
print(sorted(text))

def add3(ch):
    return chr(ord(ch)+3)
print(''.join(map(add3,text)))
print(''.join(reversed(text)))
print(list(enumerate(text)))