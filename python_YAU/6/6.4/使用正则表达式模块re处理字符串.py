#使用正则表达式模块re处理字符串
import re
import urllib.request
from string import punctuation

print(re.findall('姓名:(.+?),年龄:(\d+),性别:(.)',text))
text = "Special cases aren't special enough to break the rules."
print(re.findall('\w+',text))
print(re.split('[.\s]',text))
print(re.sub('enough','ENOUGH',text))
print(re.sub('[aes]','',text))
print(re.findall(r'\be\w+',text))
print(re.findall(r'\w+e\w+',text))
print(re.findall(r'\w+?k\b',text))
print(re.findall(r'\b\w{3}\b',text))
print(re.findall(r'\b\w{3,6}\b',text))
print(re.findall(r'\bs\w+?\b',text,re.I))
print(re.findall(r'\b\w+i\w+\b',text))
print(re.findall('\d+\.\d+\.\d+','Python 2.7.13,Python 3.6.0'))

text='''姓名:张三,年龄:26,性别:男 
姓名:王麻子,年龄:47,性别:男 
姓名:小龙女,年龄:16,性别:女 
姓名:萧峰,年龄:36,性别:男'''


text = r'one1two2three3four4five5six6seven7fight8nine9ten'
print(re.split(r'\d+',text))
print(re.findall(r'[a-zA-Z]+',text))
print(re.sub(r'\d','.',text))

print(re.sub('['+punctuation+']','','abcd,.e!f/?'))

s = "It's a very good good idea"
print(re.sub (r'(\b\w+)\1',r'\1',s))

ur1 = r'http://www.sdtbu.edu.cn/info/1123/14713.htm'
headers = {'User-Agent':'Mozilla/5.0(Windows NT 6.1; Win64 x64)AppleWebKit/537.36 (KHTML,like Gecko)Chrom/62.0.3202.62 Safari/537.36'}
req = urllib.request.Request(ur1,headers=headers)
with urllib.request.urlopen(req) as fp:
    content = fp.read().decode()
pattern = r'<img .+? src ="(.+?)"'
for imgAddress in re.findall(pattern,content):
    print(imgAddress)
