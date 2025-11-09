import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([1, 2])


def get_weight_adjust_matrix(preWeight_val, aftWeight_demands):
    plain_weights = np.full((len(a), len(b)), 1)
    weights_adjust_matrix = np.full((len(a), len(b)), 0)
    plain_weights_T = plain_weights.T
    weights_adjust_matrix += (plain_weights_T * preWeight_val).T * aftWeight_demands
    weights_adjust_matrix = weights_adjust_matrix
    return weights_adjust_matrix


def get_weight_adjust_matri1x(preWeight_val, aftWeight_demands):
    weights_adjust_matrix = np.full((len(a), len(b)), 0)
    # 外积计算每个样本的权重调整量
    weights_adjust_matrix += np.outer(preWeight_val, aftWeight_demands)
    return weights_adjust_matrix


print(get_weight_adjust_matrix(a, b))
print(get_weight_adjust_matri1x(a,b))
