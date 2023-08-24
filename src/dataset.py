import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.preprocessing import MinMaxScaler

class PatientDataset(Dataset):
    def __init__(self, data_list, tensor_list, transform=None):
        self.data_list = data_list
        self.tensor_list = tensor_list
    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        tensor = self.tensor_list[idx]
        return data, tensor


'''def create_datasets(df_clinical, df_gene, adj_mat):
    gene_list = []
    clinical_list = []
    # 识别连续数据列
    continuous_columns = df_clinical.select_dtypes(include=['float64']).columns
    # 归一化连续数据
    print(df_clinical[continuous_columns])
    scaler = MinMaxScaler()
    df_clinical[continuous_columns] = scaler.fit_transform(df_clinical[continuous_columns])
    print(df_clinical[continuous_columns])
    for idx in range(len(df_clinical.index)-1):        
        row_clinical = df_clinical.iloc[idx].drop('Sample_ID')
        sample_id = df_clinical.iloc[idx]['Sample_ID']
        edge_index = torch.tensor(np.argwhere(adj_mat == 1), dtype=torch.long).t().contiguous()
        column_gene_tensor = torch.tensor(df_gene.loc[:, sample_id], dtype=torch.float)
        print(column_gene_tensor)
        # 转换为张量
        row_clinical_tensor = torch.tensor(row_clinical.values.astype(np.float32), dtype=torch.float)
        gene_list.append(column_gene_tensor)
        clinical_list.append(row_clinical_tensor)
    graph = Data(x=torch.from_numpy(df_gene.to_numpy()).to(dtype=torch.float),edge_index = edge_index)
    print(graph)
    dataset = PatientDataset(gene_list, clinical_list)
    return dataset, graph'''

def create_datasets(df_clinical, df_gene, adj_mat,if_dataset=True):
    # 识别连续数据列
    continuous_columns = df_clinical.select_dtypes(include=['float64']).columns
    # 归一化连续数据
    scaler = MinMaxScaler()
    df_clinical[continuous_columns] = scaler.fit_transform(df_clinical[continuous_columns])

    # 获取样本ID
    sample_ids = df_clinical['Sample_ID'].tolist()

    # 删除 'Sample_ID' 列
    df_clinical = df_clinical.drop('Sample_ID', axis=1)

    # 转换为张量
    clinical_tensor = torch.tensor(df_clinical.values.astype(np.float32), dtype=torch.float)

    # 转换基因数据为张量
    gene_tensor = torch.tensor(df_gene.loc[:, sample_ids].values, dtype=torch.float).T

    # 构建图
    edge_index = torch.tensor(np.argwhere(adj_mat == 1), dtype=torch.long).t().contiguous()
    graph = Data(x=torch.from_numpy(df_gene.to_numpy()).to(dtype=torch.float), edge_index=edge_index)

    if if_dataset:
        dataset = PatientDataset(gene_tensor, clinical_tensor)
        return dataset, graph
    else:
        return gene_tensor, clinical_tensor, graph



def create_data_loaders(dataset,train_size,val_size,batch_size):
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    return train_loader, val_loader


def create_test_loader(dataset):
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    return data_loader

def create_output_loader(dataset):
    data_loader = DataLoader(dataset, batch_size=2,shuffle=False)
    return data_loader

