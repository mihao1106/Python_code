#8.1.2递归函数
#例8-2
def isPalindrome(text):
    if len(text) <= 1:
        return True
    if text[0] != text[-1]:
        return False
    return isPalindrome(text[1:-1])
sentences = ('deed','dad','need','rotor','civic','eye','redivider','noon','his','difference','a')
for sentence in sentences:
    print(sentence.ljust(12),end='')
    if isPalindrome(sentence):
        print('是回文')
    else:
        print('不是回文')