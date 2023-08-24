import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(model1,model2, val_loader, graph,device):
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (gene_data, clinical_data) in enumerate(val_loader):
            gene_data, clinical_data, graph = gene_data.to(device), clinical_data.to(device), graph.to(device)
            # 创建标签，同一患者为1，不同患者为-1
            batch_size = clinical_data.size(0)
            label = torch.eye(batch_size).to(device)
            label[label == 0] = -1 
            out1 = model1(clinical_data)
            out2 = model2(gene_data, graph)
            # 计算余弦相似度
            cos_sim = nn.functional.cosine_similarity(out1.unsqueeze(1), out2.unsqueeze(0), dim=2)
            print('Cos_sim:',cos_sim)
            # 根据阈值判断是否为同一患者
            threshold = 0.5
            pred_label = (cos_sim >= threshold).float() * 2 - 1
            print(label)
            print(pred_label)
            correct += (pred_label == label).sum().item()
            print('correct:',correct)
            print('------------------')
            total += batch_size*batch_size

    accuracy = correct / total
    return accuracy