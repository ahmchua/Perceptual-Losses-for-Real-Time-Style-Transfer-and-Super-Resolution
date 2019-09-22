import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models
import numpy as np

def train(train_params, model_params, train_loader, test_loader):
    super_resolver = model(model_params).to(args.device)
    feat = loss_net()
    feat_layer = model_params['feat_layer']

    optimizer = Adam(super_resolver.parameters(), lr = train_params['lr'])
    mse_loss = nn.MSELoss()

    for epoch in range(train_params['epochs']):
        super_resolver.train()
        epoch_loss = 0
        for batch_num, sample_x, sample_y in enumerate(train_loader):
            sample_x = sample_x.to(args.device)
            sample_y = sample_y.to(args.device)

            optimizer.zero_grad()

            pred = super_resolver(sample_x)

            f_hat = feat(pred)
            f_gold = feat(sample_y)

            C_j = f_gold.shape[0]
            H_j = f_gold.shape[1]
            W_j = f_gold.shape[2]
            loss = 1/(C_j*H_j*W_j) * mse_loss(f_hat[feat_layer], f_gold[feat_layer])
            loss = loss.to(args.device)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"epoch loss: {epoch_loss}")
    return super_resolver
