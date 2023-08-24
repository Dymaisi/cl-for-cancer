import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5, device=""):
        super().__init__()
        self.batch_size = batch_size
        mask_tensor = torch.ones(2 * batch_size, 2 * batch_size)
        # 将左上角(bs*bs)子矩阵设为0
        mask_tensor[:batch_size, :batch_size] = 0
        # 将右下角(bs*bs)子矩阵设为0
        mask_tensor[batch_size:, batch_size:] = 0
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (mask_tensor.to(device)).float())
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(z_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

def cosine_similarity_regularization(features):
    batch_size = features.size(0)

    # 计算余弦相似度
    cosine_similarity = torch.matmul(features, features.t()) / (torch.norm(features, dim=1, keepdim=True) * torch.norm(features.t(), dim=0, keepdim=True))

    # 对角线元素应为1，所以需要将它们设置为0，避免它们影响正则项
    cosine_similarity -= torch.diag(torch.diag(cosine_similarity))
    # print(cosine_similarity)
    # 计算正则项，这里我们希望最小化余弦相似度的平均值
    # 除以(batch_size * (batch_size - 1))是因为我们只考虑非对角线元素
    regularization = torch.sum(cosine_similarity.abs()) / (batch_size * (batch_size - 1))

    return regularization


