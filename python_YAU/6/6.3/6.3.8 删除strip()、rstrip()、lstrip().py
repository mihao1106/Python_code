#6.3.8 删除strip()、rstrip()、lstrip()
print('\n\nhello world  \n\n'.strip())
print("aaaassddfsdsd".strip("a"))
print("aaaassddfsdaaaasdaaaaa".rstrip("a"))  #删除右侧的指定字符
print("aaaassddfsdaaaasdaaaa".lstrip("a"))   #删除左侧的指定字符
print('aaaabbcdefgh'.strip('abcdefg'))

text = '''姓名：张三
年龄：23
性别：男
职业：学生
籍贯：地球'''
information = text.split('\n')               #删除空格
print(information)