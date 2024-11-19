#6.3.3 find()、rfind()、index()、rindex()、count()
#find()、rfind()用来检测一个字符串在整体字符串中出现的首次位置以及最后位置，不存在会返回-1
#index()、rindex()同上，不存在会报错
#count()统计一个字符串在整体字符串中出现的次数
text = 'Explicit is better than implicit.'
print(text.find("i"))
print(text.rfind("i"))
print(text.rindex("t"))
print(text.index("t"))
print(text.index("o"))
print(text.find("o"))
print(text.count("i"))