import numpy as np
from numpy.matrixlib.defmatrix import matrix

# numpy.zeros(shape, dtype=float, order='C') 默认dtype为float
# numpy.full(shape, fill_value, dtype=None, order='C', *, like=None)
# numpy.mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue, *,where=np._NoValue)

# 生成数组,说明这个数组有两行，三列，每个数字都是0
a = np.zeros((2,3))
print(a)

# 生成数组，两行三列，内容是1
b = np.full((2,3),1)
print(b)

# 把列表转化成np数组
list = [[1,2],[-3,4],[5,6]]
np_array = np.array(list)
print(np_array)


# 怎么取想要的数字,我想取第一行，第0列
index_num = np_array[1,0]
# 我想取第0列的所有数字
index_colum = np_array[:,0]
# 我想取第2行的所有内容
index_row = np_array[2,:]
print(index_num)
print(index_colum)
print(index_row)

#-------------------------------------------------
# 矩阵乘法
matrix_1 = np.array([1,2,3])
matrix_2 = np.array([3,4,5])
print(np.dot(matrix_1,matrix_2))

# Hadamard product
print(matrix_2 * matrix_1)

# add
print(matrix_2 + matrix_1)
print("-------------------------------------------------")

#-------------------------------------------------
# 求均值,axis表示方向，0表示竖着，1表示横着,默认None表示整个平均
mean = np.mean(np_array,0)
print(mean)

# 求最大值,和0比较
# 当尺寸不匹配时，maximum会尝试广播，进行匹配
print(np.maximum(0,np_array))

# 取最大值
print(np.max(np_array,1))





