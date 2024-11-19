#5.3.1集合元素增加与删除
s = {20,30,50}
print(s)
s.add(40)                #增加元素
print(s)
s.add(25)
print(s)
s.add(25)
print(s)
s.update({30,70})        #增加元素
print(s)
s.discard(50)            #删除指定元素
print(s)
s.discard(3)
print(s)
s.remove(3)
print(s)
s.pop()                  #随机弹出并删除一个元素
print(s)