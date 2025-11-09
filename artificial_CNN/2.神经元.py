import numpy as np


# activate function
def activation_ReLU(inputs):
    """
    :param inputs: matrix
    :return: matrix
    """
    return np.maximum(0, inputs)


def create_bias(n_neurons):
    """
    :param n_neurons:
    :return: bias matrix
    """
    return np.random.randn(n_neurons)


# randn = random normal 正态分布生成,双精度浮点数
# np.round 银行家舍入，采去偶舍奇进（看前一位，如1.35 = 1.4；1.25 = 1.2）
def create_weight_matrix(n_inputs, n_neurons):
    """
    :param n_inputs: 每个神经元有多少个输入
    :param n_neurons: 神经元个数
    :return: matrix
    """
    return np.random.randn(n_inputs, n_neurons)


def bank_rounding(input):
    """
    :param input:
    :return: optional
    """
    return np.round(input, 2)


# ------------------------------------------------------
a11 = -0.9
a21 = -0.4
a31 = -0.7

a12 = -0.8
a22 = -0.5
a32 = -0.6

a13 = -0.5
a23 = -0.8
a33 = -0.2
# batch
inputs = np.array([[a11, a21, a31], [a12, a22, a32], [a13, a23, a33]])

weights = bank_rounding(create_weight_matrix(3, 2))

b = bank_rounding(create_bias(2))

sum1 = np.dot(inputs, weights) + b
print(sum1)
print("--------------")
print(activation_ReLU(sum1))
