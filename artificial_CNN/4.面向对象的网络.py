import numpy as np
from twisted.web.html import output

NETWORK_SHAPE = [2, 3, 4, 2]


# activate function
def activation_ReLU(inputs):
    """
    :param inputs: matrix
    :return: matrix
    """
    return np.maximum(0, inputs)


class Layer:
    def __init__(self, n_inputs, n_neurons):
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

    def layer_forward(self, inputs):
        """
        :param inputs: 上一层的输出
        :return: matrix
        Example:
            计算单个神经元内的前向传播，并保留 3 位小数
            output = layer_forward(inputs)
        """
        sum = np.dot(inputs, self.weights) + self.biases
        self.output = activation_ReLU(sum)
        return np.round(self.output, 3)


# 定义一个网络的类
class Network:
    def __init__(self, network_shape):
        """
        :param network_shape: 每一层的神经网络的形状，如[2, 3, 4, 2]
        :return: 返回一个形状为 shape 的神经网络
        """
        # layer = shape - 1
        # 每新建一个layer，添加到layers里面
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i + 1])
            self.layers.append(layer)

    def network_forward(self, inputs):
        """
        :param inputs: 最开始的输入的 batch
        :return: outputs 会存储每一个神经元的前向传播输出结果
        """
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_outputs = self.layers[i].layer_forward(outputs[i])
            outputs.append(layer_outputs)
        return outputs


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


def main():
    network = Network(NETWORK_SHAPE)

    output = network.network_forward(inputs)
    print(network.shape)
    print(network.layers)
    print(len(network.shape))
    print(len(network.layers))
    print(output)


# ------------------------------------------------------

if __name__ == '__main__':
    ##-----------------TEST-------------------TEST------------------
    def test():
        network = Network(NETWORK_SHAPE)
        print(network.shape)
        print(network.layers)
        print(len(network.shape))
        print(len(network.layers))


    ##-----------------RUN-------------------RUN------------------
    # test()
    main()
