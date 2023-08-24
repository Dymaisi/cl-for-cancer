import torch
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from models.GCN import GCN
import scipy.io as sio
from src.dataset import create_datasets, create_output_loader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# 从CSV文件中读取数据
df_A = pd.read_csv('data/CPGEA_parsed.csv')
df_B = pd.read_csv('data/node_features_CPGEA.csv', header=0)

# 从.mat文件中读取邻接矩阵
data_dict = sio.loadmat('data/A_CPGEA_CORRELATION.mat')
adj_mat = data_dict['name']

# 识别连续数据列
continuous_columns = df_A.select_dtypes(include=['float64']).columns
# 归一化连续数据
scaler = MinMaxScaler()
df_A[continuous_columns] = scaler.fit_transform(df_A[continuous_columns])


# 设置参数s
clinical_dim = len(df_A.columns) - 1
gene_dim = len(df_B.columns)
hidden_dim_gcn = 512
hidden_dim_mlp = 128
feature_dim = 256
# batch_size = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gene_tensor, clinical_tensor, graph= create_datasets(df_A, df_B, adj_mat,if_dataset=False)
# print(gene_tensor.size())
graph = graph.to(device)
gene_tensor = gene_tensor.to(device)
model_gene = GCN(gene_dim, hidden_dim_gcn, feature_dim).to(device)
path_checkpoint = "./check_points_CPGEA/GCN_epoch_170.pt"  # 断点路径
model_gene.load_state_dict(torch.load(path_checkpoint))

# 模拟生成(1000, 100)的输入张量
# input_tensor = torch.randn(1000, 100)

output_tensor = model_gene(gene_tensor,graph)
# print(output_tensor.size)
# 把输出张量转换成numpy数组
output_array = gene_tensor.detach().cpu().numpy()
# print(output_array)
# print(output_array.shape[0])
# print(output_array.shape[1])
# 创建一个Pandas DataFrame，包括生存时间和事件发生状态
# 假设你已经有了包含生存时间（'duration'）和事件发生状态（'event'）的数据
file_path = 'data/CPGEA_parsed_cox.csv'

# 从CSV文件中导入前两列
survival_data = pd.read_csv(file_path, usecols=[1, 2])




# 把神经网络的输出添加到数据集中

for i in range(output_array.shape[1]):
    survival_data[f"feature_{i}"] = output_array[:, i]


# 假设df是一个Pandas DataFrame对象
survival_data.to_csv('survival_data_cpgea.csv', encoding='utf-8')


# print(survival_data)
# 实例化一个CoxPHFitter对象
cph = CoxPHFitter()

# 使用数据拟合Cox-PH模型
cph.fit(survival_data, duration_col='time', event_col='dead')
# cph.plot()
cph.print_summary() 
# plt.show()

# 计算患者的Risk Score
# Risk Score = sum(feature_i * feature_i_coef)
risk_scores = np.zeros(survival_data.shape[0])
for i in range(output_array.shape[1]):
    feature_coef = cph.params_[f"feature_{i}"]
    risk_scores += survival_data[f"feature_{i}"] * feature_coef

print("Risk Scores:")
print(risk_scores)

# 计算C-Index
c_index = concordance_index(survival_data['time'], -risk_scores, survival_data['dead'])
print("C-Index:", c_index)
