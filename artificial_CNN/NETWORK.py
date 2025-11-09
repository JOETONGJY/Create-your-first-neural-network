import numpy as np
import copy
import math
import sys

import artificial_CNN.data_create as c_data

NETWORK_SHAPE = [2, 10, 20, 15, 10, 2]
BATCH_SIZE = 500
LEARNING_RATE = 0.015

force_train = False
random_train = False
n_improved = 0
n_not_improved = 0


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


def loss_func2(predicted, real):
    """
    损失函数，如果两个向量的方向越接近，两个向量相乘的dot越接近1
    根据loss的定义，返回 1 - product ，表示越小越好
    :param predicted: 预测向量矩阵
    :param real: 计算出来的向量矩阵
    :return: loss，越小越接近
    """
    condition = predicted > 0.5
    binary_predicted = np.where(condition, 1, 0)
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(binary_predicted * real_matrix, axis=1)
    return 1 - product


# 需求函数，最后的值应该怎么变，为反向传播铺垫
def get_final_layer_preAct_demands(predicted_val, target_vector):
    """
    反向传播需求函数
    :param predicted_val: 预测值，如[0.3,0.7],[1,0]
    :param target_vector: 目标矩阵，二分类，所以大小是（预测值的长度的行数） * 2列
    :return: 大小为target_vector的矩阵，1表示应该增大，-1应该减小
    """
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector
    for i in range(len(target_vector)):
        if np.dot(target[i], predicted_val[i]) > 0.5:
            target[i] = np.array([0, 0])
        else:
            target[i] = (target[i] - 0.5) * 2
    return target


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


def vector_normalize(array):
    """
    标注化函数，[10,2] -> [1,0.2] ; [3,-6] -> [0.5,-1]
    :param array:需要标准化的数组
    :return: 标准化后的数组
    """
    max = np.max(np.absolute(array))
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
        :Example:
            计算单个神经元内的前向传播，并保留 3 位小数
            output = layer_forward(inputs)
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def layer_backward(self, afterWeights_demands, preWeights_values):
        """
        反向传播的层的传播过程

        :parameter---------------------------
        :param afterWeights_demands: 输出层的 demand
        :param preWeights_values: 输入层的 value
        :param condition: 这个分支的作用在于判断前一个神经元是否被ReLu激活，如果False说明小于0

        :returns-----------------------------
        :return: 当前层的调整矩阵，以及上一层输入层的 demands
        """
        preWeights_demands = np.dot(afterWeights_demands, self.weights.T)

        condition = preWeights_values > 0
        value_derivatives = np.where(condition, 1, 0)
        preActs_demands = value_derivatives * preWeights_demands
        norm_preActs_demands = normalize(preActs_demands)

        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, afterWeights_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)

        return norm_preActs_demands, norm_weight_adjust_matrix

    # 计算每个样本的梯度，并在最后求平均
    def get_weight_adjust_matrix(self, preWeight_val, aftWeight_demands):
        """
        调整矩阵
        :param preWeight_val: 输入的矩阵，对应输入维度
        :param aftWeight_demands: 损失信号矩阵，对应输出神经元的维度
        :return: 一个形状和 weights一模一样的矩阵，内容是两者的数量积
        """
        weights_adjust_matrix = np.zeros(self.weights.shape)
        for i in range(BATCH_SIZE):
            # 外积计算每个样本的权重调整量
            weights_adjust_matrix += np.outer(preWeight_val[i, :], aftWeight_demands[i, :])
        weights_adjust_matrix /= BATCH_SIZE
        return weights_adjust_matrix


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

    def network_backward(self, layer_outputs, target_vector):
        """
         网络的反向传播
        :param layer_outputs: 正向传播后的最后一层的结果
        :param target_vector: 目标矩阵，可以是一个标准答案的矩阵
        :return: 一个新的备用网络
        """
        backup_network = copy.deepcopy(self)  # 备用网络
        preAct_demands = get_final_layer_preAct_demands(layer_outputs[-1], target_vector)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - i - 1]
            if i != 0:
                layer.biases += np.mean(preAct_demands, axis=0) * LEARNING_RATE
                layer.biases = vector_normalize(layer.biases)  # 偏置更新

            outputs = layer_outputs[len(layer_outputs) - i - 2]
            results_list = layer.layer_backward(preAct_demands, outputs)
            preAct_demands = results_list[0]
            weights_adjust_matrix = results_list[1]
            layer.weights += LEARNING_RATE * weights_adjust_matrix  # 权重更新
            layer.weights = normalize(layer.weights)

        return backup_network

    def one_batch_train(self, batch):
        """
        单个batch训练
        :param batch: 一次性输入训练的个数
        :return: None
        """
        global force_train, random_train, n_improved, n_not_improved

        inputs = (batch[:, :2])
        target = copy.deepcopy(batch[:, 2]).astype(int)  # 标准答案
        outputs = self.network_forward(inputs)
        precise_loss = loss_func(outputs[-1], target)
        loss = loss_func2(outputs[-1], target)

        if np.mean(precise_loss) <= 0.05:
            print("达到一定水平，不用训练")
        else:
            backup_network = self.network_backward(outputs, target)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = loss_func(backup_outputs[-1], target)
            backup_loss = loss_func2(backup_outputs[-1], target)

            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                n_improved += 1
                # print("IMPROVED")
            else:
                n_not_improved += 1
                # print("None")
        print("分割——————————————————")

    def train(self, n_entries):
        """
        多批次训练，调用了单批次的接口
        :param n_entries: 一共有多少数据，然后除以一个BATCH_SIZE就是每一次训练的数据量
        :return: 训练完的模型，或许还有可视化
        """
        global force_train, random_train, n_improved, n_not_improved
        n_improved = 0
        n_not_improved = 0

        n_batches = math.ceil(n_entries / BATCH_SIZE)
        for i in range(n_batches):
            batch = c_data.create_data(BATCH_SIZE)
            self.one_batch_train(batch)
        improvement_rate = n_improved / (n_improved + n_not_improved)
        print("Improvement_rate:")
        print(format(improvement_rate, ".0%"))
        data = c_data.create_data(BATCH_SIZE)
        c_data.show_data(data, "right classify")
        inputs = data[:, :2]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        c_data.show_data(data, "after training")

    # ------------------------------------------------------


def main():
    # 初始化数据
    data = c_data.create_data(BATCH_SIZE)  # 生成数据

    use_this_network = 'n'  # n 代表不用
    # 选择起始网络
    while use_this_network != 'Y':
        network = Network(NETWORK_SHAPE)
        inputs = data[:, :2]
        outputs = network.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        c_data.show_data(data, "NETWORK you choose")
        use_this_network = input("Use this net?Y/n:\n")

    do_train = input("Train?Y/n,or just type your n_entries:\n")
    while do_train == 'Y' or 'y' or do_train.isnumeric() == True:
        if do_train.isnumeric():
            n_entries = int(do_train)
        else:
            n_entries = int(input("type your n_entries:\n"))

        network.train(n_entries)
        save_text = input("You want to save this result?Y/n:\n")
        if save_text == "Y":
            print("\nThese are the layer parameters:")
            for i, layer in enumerate(network.layers):
                print(f"\nLayer {i}:")
                print("weights shape:", layer.weights.shape)
                print(layer.weights)
                print("biases shape:", layer.biases.shape)
                print(layer.biases)
        else:
            pass

        do_train = input("Train?Y/n:\n")
        if do_train == "n":
            sys.exit()
        else:
            pass

    # 训练效果
    inputs = data[:, :2]
    outputs = network.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:, 2] = classification
    c_data.show_data(data, "After training")

    # c_data.show_data(data, "Correct")
    # print(data)

    # ------------------------------------------------------

    # 构建网络
    # network = Network(NETWORK_SHAPE)
    #
    # inputs = data[:, :2]
    # outputs = network.network_forward(inputs)
    # classification = classify(outputs[-1])
    # data[:, 2] = classification
    # c_data.show_data(data, "before training")
    #
    # n_entries = int(input("Enter the number fo data entries used to train\n"))
    # network.train(n_entries)

    # print("输出分割线——————————————————")
    # # 输出预测结果classification
    # classification = classify(output[-1])
    #
    # # 把softmax之后的值覆盖data第三列，输出
    # data[:, 2] = classification
    # print(data)
    # c_data.show_data(data, "Before")
    #
    # backup_network = network.network_backward(output, target)
    # new_output = backup_network.network_forward(inputs)
    # new_classification = classify(new_output[-1])
    # data[:, 2] = new_classification
    # c_data.show_data(data, "After")

    # # loss
    # loss = loss_func(output[-1], target)
    # print(loss)
    #
    # demands = get_final_layer_preAct_demands(output[-1], target)
    # print("output")
    # print(output[-1])
    # print(demands)
    # # c_data.show_data(data, "after")
    #
    # print("调整矩阵——————————————————")
    # adjust_matrix = network.layers[-1].get_weight_adjust_matrix(output[-2], demands)
    # print(adjust_matrix)
    #
    # layer_backward = network.layers[-1].layer_backward(demands, output[-2])
    # print(layer_backward)


# ------------------------------------------------------

if __name__ == '__main__':
    main()
