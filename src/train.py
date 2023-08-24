import torch
import os
from .contrastive_loss import cosine_similarity_regularization
from .evaluate import evaluate
import torchvision
from torch.utils.tensorboard import SummaryWriter
def train(model_clinical,model_gene, dataloader, val_loader, optimizer_clinical, optimizer_gene, num_epochs, loss_fn,graph, device, save_path,resume = False):
    writer = SummaryWriter('runs/64')
    model_clinical.train()
    model_gene.train()
    loss_values = []
    val_accuracys = []
    start_epoch = -1
    if resume:
        path_checkpoint1 = "./saved_model/MLP_epoch_135.pt"  # 断点路径
        path_checkpoint2 = "./saved_model/GCN_epoch_135.pt"
        model_clinical.load_state_dict(torch.load(path_checkpoint1))  # 加载断点
        model_gene.load_state_dict(torch.load(path_checkpoint2))
        # optimizer_clinical.load_state_dict(model_clinical_checkpoint['optimizer'])  # 加载优化器参数
        # optimizer_gene.load_state_dict(model_gene_checkpoint['optimizer'])
        # start_epoch = model_clinical.load_state_dict(torch.load(path_checkpoint1))[2]  # 设置开始的epoch
        # print(start_epoch)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            gene_data, clinical_data = batch
            gene_data, clinical_data, graph = gene_data.to(device), clinical_data.to(device), graph.to(device)   
            out1 = model_clinical(clinical_data)
            out2 = model_gene(gene_data, graph)

            optimizer_gene.zero_grad()
            optimizer_clinical.zero_grad()
            # loss1 = cosine_similarity_regularization(out2)
            loss2 = loss_fn(out1, out2)
            # print('Clinical_cos_sim:',loss1.item())
            print('NTXent_loss:',loss2.item())
            running_loss += loss2.item()
            loss2.backward()
            # loss2.backward(retain_graph=True)
            # loss1.backward()
            optimizer_gene.step()
            optimizer_clinical.step()
        torch.save(model_clinical.state_dict(), os.path.join(save_path, f'MLP_epoch_{epoch+1}.pt'))
        torch.save(model_gene.state_dict(), os.path.join(save_path, f'GCN_epoch_{epoch+1}.pt'))
        writer.add_scalar('training loss', running_loss / len(dataloader), epoch)
        val_accuracy = evaluate(model_clinical, model_gene, val_loader, graph, device)
        writer.add_scalar('validation accuracy', val_accuracy, epoch)
        loss_values.append(loss2.item())
        val_accuracys.append(val_accuracy)
    return loss_values,val_accuracys
