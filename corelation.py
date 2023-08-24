import pandas as pd
import numpy as np
import scipy.io as io
# 读取xlsx文件到一个DataFrame对象
df = pd.read_csv('MyCL/data/node_features_TCGA.csv', delimiter=',', header=0)

# 计算相关系数并存储在矩阵中
print(df)
correlation_matrix = np.corrcoef(df)

# 初始化零矩阵
A = np.zeros(correlation_matrix.shape)

# 遍历相关系数矩阵来判断是否填写1
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        if correlation_matrix[i,j] > 0.55:
            A[i,j] = A[j,i] = 1

# 输出结果

D = np.diag(np.sum(A, axis=1))

io.savemat('MyCL/data/A_TCGA_CORRELATION.mat', {'name': A})
io.savemat('MyCL/data/D_TCGA_CORRELATION.mat', {'name': D})

