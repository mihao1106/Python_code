# 例8-7 账户余额
def balance(base, rate):
    while True:
        base += base * rate
        yield base


base = 10
rate = 0.02
for year, current in enumerate(balance(base, rate), start=1):
    if current >= 2 * base:
        print(year, current)
        break
