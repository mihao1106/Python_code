#6.5.1 分词  pip 安装扩展库
import jieba
text = 'Python之禅中有句话非常重要,Readability counts.'
print(jieba.lcut(text))

text = "Python之禅中有句话非常重要,Readability counts."
from pypinyin import lazy_pinyin
print(lazy_pinyin)