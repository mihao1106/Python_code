#例6-2
from random import choice
from string import ascii_letters,digits
characters = digits + ascii_letters
def generatePassword(n):
    return ''.join((choice(characters) for _ in range(n)))
print(generatePassword(8))
print(generatePassword(15))