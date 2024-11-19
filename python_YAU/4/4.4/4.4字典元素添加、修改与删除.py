#4.4字典元素添加、修改与删除
sock = {'IP':'127.2.3','port':80}
sock['port'] = 90           #更改字典中的元素
sock['portocol'] = 'TCP'    #在字典中加入新的元素
print(sock)

sock = {'IP':'127.2.3','port':80}
sock.update({'IP':'190.4.6','portocol':'TCP'})
print(sock)

print(sock.popitem())        #随机删除一个元素
print(sock.pop('IP'))

del sock['port']
print(sock)