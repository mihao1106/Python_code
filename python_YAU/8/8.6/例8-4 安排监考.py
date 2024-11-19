# 例8-4 安排监考(运行的结果与课本上差别很大)
from random import shuffle  # 随机


def func(teacherNames, examNumbers, maxPerTeacher):
    '''teacherNames:
    examNumbers:
    maxPerTeacher:
    '''
    #
    teachers = {teacher: 0 for teacher in teacherNames}
    result = []
    for _ in range(examNumbers):
        teacher1 = min(teachers.items(), key=lambda item: item[1])[0]
        restTeachers = [item for item in teachers.items() if item[0] != teacher1]
        shuffle(restTeachers)
        teacher2 = min(restTeachers, key=lambda item: item[1])[0]
        if max(teachers[teacher1], teachers[teacher2]) >= maxPerTeacher:
            return '数据不合适'
            teachers[teacher1] += 1
            teachers[teacher2] += 1
            result.append((teacher1, teacher2))
        return result
    teacherNames = ['教师' + str(i) for i in range(10)]
    result = func(teacherNames, 32, 10)
    print(result)
    if result != '数据不合适':
        for teacher in teacherNames:
            num = sum(1 for item in result if teacher in item)
            print(teacher, num)
