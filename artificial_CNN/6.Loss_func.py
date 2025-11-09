import numpy as np
from numba.experimental.jitclass.overloads import class_int

import artificial_CNN.data_create as c_data

NETWORK_SHAPE = [2, 3, 4, 2]


# Relu激活函数
def activation_ReLU(inputs):
    """
    RELU函数的构造
    :param inputs: matrix
    :return: matrix
    """
    return np.maximum(0, inputs)


# SoftMax函数
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


# 损失函数
def loss_func(predicted, real):
    """
    损失函数，如果两个向量的方向越接近，两个向量相乘的dot越接近1
    根据loss的定义，返回 1 - product ，表示越小越好
    :param predicted: 预测向量矩阵
    :param real: 计算出来的向量矩阵
    :return: loss，越小越接近
    """
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted * real_matrix, axis=1)
    return 1 - product


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
    return np.rint(probabilities[:, 1])


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
    # 初始化数据
    data = c_data.create_data(10)
    inputs = data[:, :2]
    print(data)

    # target取data的tag，代表标准答案
    target = data[:, 2]

    c_data.show_data(data, "before")

    # 构建网络
    network = Network(NETWORK_SHAPE)
    output = network.network_forward(inputs)

    # 输出预测结果classification
    classification = classify(output[-1])
    print(classification)

    # 把softmax之后的值覆盖data第三列，输出
    data[:, 2] = classification
    print(data)

    # loss
    loss = loss_func(output[-1], target)
    print(loss)

    c_data.show_data(data, "after")


# ------------------------------------------------------

if __name__ == '__main__':
    main()
