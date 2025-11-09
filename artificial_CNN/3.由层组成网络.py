import numpy as np


# 希望构建：input 2 , output 2 , 总层数 3
# layer1:3结点，layer2:4结点，layer3(output):2结点

# activate function
def activation_ReLU(inputs):
    """
    :param inputs: matrix
    :return: matrix
    """
    return np.maximum(0, inputs)

class Layer:
    def __init__(self,n_inputs,n_neurons):
        """
        :param n_inputs: 输入的维度
        :param n_neurons: 神经元的数量
        :return: 权重矩阵 与 bias
        Example:
            输入维度为 2， 下一层的神经元有 3 个
            Layer1 = Layer(2, 3)
        """
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(n_neurons)

    def layer_forward(self,inputs):
        """
        :param inputs: 上一层的输出
        :return: matrix
        Example:
            计算前向传播，并保留 3 位小数
            output = layer_forward(inputs)
        """
        sum = np.dot(inputs, self.weights) + self.biases
        self.output = activation_ReLU(sum)
        return np.round(self.output,3)

# ------------------------------------------------------
a11 = -0.9
a21 = 0.4

a12 = 0.8
a22 = -0.5

a13 = -0.5
a23 = -0.8

a14 = 0.2
a24 = -0.3

a15 = -0.6
a25 = 0.5
# batch
inputs = np.array([[a11, a21], [a12, a22], [a13, a23], [a14, a24], [a15, a25]])

# ------------------------------------------------------

# layer1
Layer1 = Layer(2,3)

# layer2
Layer2 = Layer(3,4)

# layer3
Layer3 = Layer(4,2)

# calculate_layer1
output1 = Layer1.layer_forward(inputs)

# calculate_layer2
output2 = Layer2.layer_forward(output1)

# calculate_layer3
output3 = Layer3.layer_forward(output2)

print("--------------")
print(activation_ReLU(output3))
