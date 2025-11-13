import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算

print(torch.exp(x))  # “按元素”方式可以应用更多的计算，包括像求幂符

print("------------------------")

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 沿行接（dim0），行数是原先的两倍；沿列接（dim1），列数是原先的两倍
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

print("-------------------------")

print(X == Y)  # 我们想通过逻辑运算符构建二元张量
print(X.sum())  # 对张量中的所有元素进行求和，会产生一个单元素张量
