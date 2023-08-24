import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, gene_dim, hidden_dim, feature_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(gene_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, feature_dim)

    def forward(self, data, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        out = torch.matmul(data, x)
        return out
    
'''
class GCN_for_Cox(torch.nn.Module):
    def __init__(self, gene_dim, hidden_dim, feature_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(gene_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, feature_dim)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x'''