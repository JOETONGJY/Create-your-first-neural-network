import matplotlib.pyplot as plt
import numpy as np
import math
import random
import matplotlib.pyplot as plot

NUM_OF_DATA = 1000


def tag(x, y):
    """
    判断是否在圈内
    :param x: 横坐标
    :param y: 纵坐标
    :return: 在圈内，tag = 1，反之 tag = 0
    """
    # return x ** 2 + y ** 2 < 1
    return x >= 0 and y >= 0 or x <= 0 and y <= 0 # 四方格


def create_data(num_of_data):
    """
    生成随机数
    :param num_of_data: 需要生成的组的数量
    :return: 数据
    """
    entry_list = []
    for i in range(num_of_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        entry = [x, y, tag(x, y)]
        entry_list.append(entry)
    return np.array(entry_list)


def show_data(data, title):
    color = []
    for tag in data[:, 2]:
        color.append("orange" if tag else "blue")
    plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.title(title)
    plt.show()

# ------------------------------------------------
