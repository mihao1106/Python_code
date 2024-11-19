#8.1.1基本语法
#例8-1
def getFrequency(s):
    digits,alphabets,others = 0,0,0
    for ch in s:
        if '0'<=ch<='9':
            digits += 1
        elif 'a'<=ch<='z' or 'A'<=ch<='Z':
            alphabets += 1
        else:
            others += 1
    return (digits,alphabets,others)
print(getFrequency('123,./abc'))