#例7-1  2018年第47周周四是2018年11月22日
from datetime import date,timedelta
year,n,w = map(int,input('请输入year n w :').split())
start = date(year,1,1)
for i in range(7):
    if start.isoweekday()==w:
        break
    start = start + timedelta(days=1)
print(start  + timedelta(weeks=n-1))