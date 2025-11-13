import torch

x = torch.arange(12)  # 我们可以使用 arange 创建一个行向量 x

print(x.shape)  # 可以通过张量的shape属性来访问张量（沿每个轴的长度）的形状
print(x.numel())  # 如果只想知道张量中元素的总数，即形状的所有元素乘积，可以检查它的大小（size）

X1 = x.reshape(3, 4)  # 要想改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数
X2 = x.reshape(3, -1)  # 在知道宽度后，高度会被自动计算得出，不必我们自己做除法
print(X1.shape == X2.shape)

print("____________________")

x = torch.zeros((2, 3, 4))  # 有时，我们希望使用全0、全1、其他常量,或是随机
x = torch.ones((2, 3, 4))
print(x)

print("____________________")

x = torch.randn(3, 4) # 均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]) # 提供包含数值的Python列表
print(x)
