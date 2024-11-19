#6.3.2 encode() 使用指定的编码格式把字符串编码改为字节串
bookName = '《Python可以这样学》'
print(bookName.encode())
print(bookName.encode('gbk'))
bookName = '《Python程序设计开发宝典》'
print(bookName.encode().decode('gbk'))