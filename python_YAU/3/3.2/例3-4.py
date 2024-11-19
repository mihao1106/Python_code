#例3—4
from random import sample
matrix=[sample(range(1,20),8) for i in range(5)]
for row in matrix:
    print(row)

result = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
for row in result:
    print(row)

result = []
for i in range(len(matrix[0])):
    newRow=[]
    for row in matrix:
        newRow.append(row[i])
    result.append(newRow)
for row in result:
    print(row)

import numpy as np
matrix = np.matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(matrix)
print(matrix.T)

matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
for row in matrix:
    print(row)

result = list(map(list,zip(*matrix)))
for row in result:
    print(row)