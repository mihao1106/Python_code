#6.3.5 join()将多个字符串相连接
print(':'.join(map(str,range(10))))
print(','.join('abcdefg'.split('cd')))
print('_'.join(['1','2','3','4']))
text = 'How old are you?'
print(' '.join(text.split( )))