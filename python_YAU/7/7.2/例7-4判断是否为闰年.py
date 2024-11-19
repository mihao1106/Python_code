#例7-4判断是否为闰年
try:
    year = int(input('请输入一个年份：'))
except:
    print('输入错误，请输入表示年份的整数。')
else:                                                    #方法1
    if year%400==0 or (year%4==0 and year%100!=0):
        print('Yes')
    else:
        print('No')

year = int(input('请输入一个年份：'))                      #自我检测
if year%400==0 or (year%4==0 and year%100!=0):
    print(f'您输入的年份是：{year},是闰年')
else:
    print(f'您输入的年份是：{year},不是闰年')

import calendar                                           #方法2
print(calendar.isleap(2016))
print(calendar.isleap(3456))
print(calendar.isleap(2000))