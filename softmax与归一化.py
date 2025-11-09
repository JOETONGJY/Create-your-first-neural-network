import numpy as np

# softmax
# 把py数组转化为np数组
x = np.array([2.0,1.0,0.1])
# 对数组中的每个元素执行指数运算
softmax = np.exp(x) / np.sum(np.exp(x))
print(softmax)

# tips
# 为了防止指数函数爆炸，通常会减去最大值做数值稳定处理
softmax_better = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
print(softmax_better)

# norm,min-max归一化
norm = (x - np.min(x)) / (np.max(x) - np.min(x))
print(norm)