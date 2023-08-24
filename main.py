import pandas as pd
import scipy.io as sio
import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.dataset import create_datasets,create_data_loaders
from src.train import train
from src.evaluate import evaluate
from src.contrastive_loss import NTXentLoss
from models.GCN import GCN
from models.MLP import MLP

# 准备数据
# 从CSV文件中读取数据
df_A = pd.read_csv('data/TCGA_parsed.csv')
df_B = pd.read_csv('data/node_features_TCGA.csv', header=0)

# 从.mat文件中读取邻接矩阵
data_dict = sio.loadmat('data/A_TCGA_CORRELATION.mat')
adj_mat = data_dict['name']

# 设置参数
clinical_dim = len(df_A.columns) - 1
gene_dim = len(df_B.columns)
hidden_dim_gcn = 32
# hidden_dim2_gcn = 64
hidden_dim_mlp = 32
feature_dim = 128
num_epochs = 64
batch_size = 4
learning_rate = 0.0001

dataset,graph = create_datasets(df_A, df_B, adj_mat)
print(len(dataset))
train_size = 352
val_size = 121
# train_size = 100
# val_size = 26



train_loader,val_loader = create_data_loaders(dataset,train_size,val_size,batch_size)
# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_clinical = MLP(clinical_dim, hidden_dim_mlp, feature_dim).to(device)
model_gene = GCN(gene_dim, hidden_dim_gcn, feature_dim).to(device)
loss_fn = NTXentLoss(batch_size=batch_size, device=device)
# alpha = 0.5
save_path = "check_points"
# 优化器
optimizer_clinical = optim.Adam(model_clinical.parameters(), lr=learning_rate)
optimizer_gene = optim.Adam(model_gene.parameters(), lr=learning_rate)
# 训练模型
loss_values,val_accuracys = train(model_clinical, model_gene, train_loader,val_loader, optimizer_clinical, optimizer_gene, num_epochs, loss_fn, graph, device, save_path)

# 绘制损失曲线
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 绘制验证准确曲线
plt.plot(val_accuracys)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.show()

min_value, min_index = min((val, idx) for idx, val in enumerate(loss_values))

print("best_epoch_loss:", min_value)
print("best_epoch:",min_index)
# 加载模型
model_clinical.load_state_dict(torch.load(os.path.join(save_path, f'MLP_epoch_{min_index}.pt')))
model_gene.load_state_dict(torch.load(os.path.join(save_path, f'GCN_epoch_{min_index}.pt')))
# 验证模型
val_accuracy = evaluate(model_clinical, model_gene, val_loader, graph, device)
print('Validation Accuracy: {:.4f}'.format(val_accuracy))