#6.3.15数据压缩与解压缩
import zlib
text = '《Python程序设计（第2版）》、《Python程序设计基础（第2版）》、董付国编著'.encode()
print(text)
print(len(text))

y = zlib.compress(text)
print(y)
print(len(y))
print(zlib.decompress(y).decode())