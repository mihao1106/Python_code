#6.3.10 isalnum()、isalpha()等
print('1234abcd'.isalnum())   #测试是否只包含数字与字母
print('\t\n\r'.isspace())     #测试是否全部为空白字符
print('aBc'.isupper())        #测试是否全部为大写字母
print('1234abcd'.isalpha())   #测试是否全部为英文
print('1234abcd'.isdigit())   #测试是否全部为数字
print('1234.0'.isdigit())     #不能测试浮点数
print('1234'.isdigit())       #只能测试整数