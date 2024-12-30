'''
问题：
对于函数z=x²+y²，请用梯度下降法求出使函数取得最小值的x、y值

过程：
使用步长（学习率）为0.1，初始位置为x=3.0,y=2.0,
通过计算函数关于x和y的偏导数来找到梯度，然后沿着梯度的反方向更新x和y的值
'''


def gradient_descent(learning_rate=0.1, num_iterations=1000):
    # 初始点
    x, y = 3.0, 2.0

    for i in range(num_iterations):
        grad_x = 2 * x  # 对x的偏导数
        grad_y = 2 * y  # 对y的偏导数

        # 更新x和y的值
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y

        # 打印每次迭代的x和y值
        if i % 10 == 0:  # 每10次迭代打印一次，以减少输出量
            print(f"Iteration {i + 1}: x = {x:.2f}, y = {y:.2f}")

    return x, y


# 调用梯度下降函数
min_x, min_y = gradient_descent()
print(f"Minimum point: x = {min_x:.2f}, y = {min_y:.2f}")
