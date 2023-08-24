import torch
import os
import pandas as pd
import scipy.io as sio
from src.dataset import create_datasets, create_output_loader
from src.evaluate import evaluate
from models.GCN import GCN
from models.MLP import MLP


# 从CSV文件中读取数据
df_A = pd.read_csv('data/CPGEA_parsed.csv')
df_B = pd.read_csv('data/node_features_CPGEA.csv', header=0)

# 从.mat文件中读取邻接矩阵
data_dict = sio.loadmat('data/A_CPGEA_CORRELATION.mat')
adj_mat = data_dict['name']

# 设置参数s
clinical_dim = len(df_A.columns) - 1
gene_dim = len(df_B.columns)
hidden_dim_gcn = 512
hidden_dim_mlp = 128
feature_dim = 256
batch_size = 16

dataset,graph = create_datasets(df_A, df_B, adj_mat)
test_loader =  create_output_loader(dataset)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = MLP(clinical_dim, hidden_dim_gcn, feature_dim).to(device)
model2 = GCN(gene_dim, hidden_dim_mlp, feature_dim).to(device)

save_path = "check_points_CPGEA"
# 加载模型
model1.load_state_dict(torch.load(os.path.join(save_path, f'MLP_epoch_170.pt')))
model2.load_state_dict(torch.load(os.path.join(save_path, f'GCN_epoch_170.pt')))


for data, tensor in dataset:
    data,tensor = data.to(device),tensor.to(device)
    graph = graph.to(device)
    print(tensor)
    output1 = model1(tensor)
    output2 = model2(data,graph)
    print(output1,output2)
    break