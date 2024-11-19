#导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random

#假设数据文件包含这些列，加载数据
data = pd.read_csv('/path/to/your/data.csv')

# 提取相关特征
features = ['phy_rate', 'loss_rate', 'interference']
target = 'throughput'

# 进行特征缩放
scaler = StandardScaler()
X = scaler.fit_transform(data[features])

# 初始化吞吐量函数所需的数据
phy_rate = data['phy_rate'].values  # PHY速率
loss_rate = data['loss_rate'].values  # 丢包率
interference = data['interference'].values  # 干扰强度
n = len(phy_rate)  # AP数量

# 定义系统吞吐量的计算公式
def calc_throughput(t, phy_rate, loss_rate):
    throughput = np.sum(phy_rate * t * (1 - loss_rate))
    return throughput

# 创建遗传算法的FitnessMax和Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 遗传算法工具箱
toolbox = base.Toolbox()

# 初始化个体
toolbox.register("attr_float", random.random)

# 定义个体生成与种群生成
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评估函数：计算吞吐量
def evaluate(individual):
    t = np.array(individual)  # 个体t_i
    t = t / np.sum(t)  # 确保t_i的和为1
    throughput = calc_throughput(t, phy_rate, loss_rate)
    return throughput,

### 注册遗传算法操作
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

### 运行遗传算法
def run_ga():
    pop = toolbox.population(n=100)  # 初始种群
    hof = tools.HallOfFame(1)  # 保存最优个体
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.2, ngen=50, verbose=True)
    return pop, logbook, hof

### 运行遗传算法
pop, logbook, hof = run_ga()

### 输出最优解
best_ind = hof[0]
best_throughput = best_ind.fitness.values[0]
print(f"最优的发送机会分配: {best_ind}")
print(f"最大系统吞吐量: {best_throughput}")

### 绘制吞吐量的进化过程
gen = logbook.select("gen")
max_fitness = logbook.select("max")
avg_fitness = logbook.select("avg")

### plt.figure(figsize=(10, 6))
plt.plot(gen, max_fitness, label="最大适应度", color='b')
plt.plot(gen, avg_fitness, label="平均适应度", color='g')
plt.xlabel("代数")
plt.ylabel("系统吞吐量")
plt.title("遗传算法优化吞吐量的进化过程")
plt.legend()

# 绘制图形
plt.grid(True)
plt.show()

# 绘制最优的发送机会分配
plt.figure(figsize=(10, 6))
plt.bar(range(n), best_ind, color='c')
plt.xlabel("AP编号")
plt.ylabel("发送机会分配")
plt.title("最优发送机会分配")
plt.grid(True)
plt.show()