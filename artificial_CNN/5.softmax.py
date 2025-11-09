import numpy as np
from numba.experimental.jitclass.overloads import class_int

import artificial_CNN.data_create as c_data

NETWORK_SHAPE = [2, 3, 4, 2]


# activate function
def activation_ReLU(inputs):
    """
    RELU函数的构造
    :param inputs: matrix
    :return: matrix
    """
    return np.maximum(0, inputs)


def activation_softmax(inputs):
    """
    SoftMax的构造
    :param inputs:
    :return:
    """
    # 取每一行的最大值
    max_val = np.max(inputs, 1, keepdims=True)
    # 防止指数过大
    slided_inputs = inputs - max_val
    # 求指数
    exp_val = np.exp(slided_inputs)
    # 把每一行求和，然后相除
    sum = np.sum(exp_val, 1, keepdims=True)
    return exp_val / sum


# def where(condition, x=None, y=None) 通常用于数组矢量化条件选择
def normalize(array):
    """
    标注化函数，[10,2] -> [1,0.2] ; [3,-6] -> [0.5,-1]
    :param array:需要标准化的数组
    :return: 标准化后的数组
    """
    max = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max == 0, 0, 1 / max)
    return array * scale_rate

# rint 即 round int 是舍入到整数位 的 np.round
def classify(probabilities):
    """
    银行家舍入到整数位
    :param probabilities: 归一化后的概率,是一个数字
    :return:类别 0/1
    """
    return np.rint(probabilities[:,1])


class Layer:
    def __init__(self, n_inputs, n_neurons):
        """
        构造Layer
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
        前向传播,结果不经过非线性处理
        :param inputs: 上一层的输出
        :return: matrix
        Example:
            计算单个神经元内的前向传播，并保留 3 位小数
            output = layer_forward(inputs)
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


# 定义一个网络的类
class Network:
    def __init__(self, network_shape):
        """
        构造Network
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
        控制layer前向传播
        :param inputs: 最开始的输入的 batch
        :return: outputs 会存储每一个神经元的前向传播输出结果
        """
        outputs = [inputs]
        for i in range(len(self.layers)):
            temp = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers) - 1:
                layer_outputs = activation_ReLU(temp)
            else:
                layer_outputs = activation_softmax(temp)
            outputs.append(layer_outputs)
        return outputs


# ------------------------------------------------------

def main():
    data = c_data.create_data(50)
    inputs = data[:, :2]
    print(data)
    c_data.show_data(data, "before")
    network = Network(NETWORK_SHAPE)
    output = network.network_forward(inputs)
    classification = classify(output[-1])
    data[:,2] = classification
    print(data)
    c_data.show_data(data,"after")



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
