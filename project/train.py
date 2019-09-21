import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models
import numpy as np

def train(train_params, model_params, train_loader, test_loader):
    super_resolver = model(model_params)
    feat = loss_net()
    feat_layer = model_params['feat_layer']

    optimizer = Adam(enc_model.parameters(), lr = train_params['lr'])
    mse_loss = nn.MSELoss()

    for epoch in range(train_params['epochs']):
        super_resolver.train()
        epoch_loss = 0
        for batch_num, samples in enumerate(train_loader):
            optimizer.zero_grad()

            pred = super_resolver(samples['x'])

            f_hat = feat(pred)
            f_gold = feat(samples['y'])

            C_j = f_gold.shape[0]
            H_j = f_gold.shape[1]
            W_j = f_gold.shape[2]
            loss = 1/(C_j*H_j*W_j) * mse_loss(f_hat[feat_layer], f_gold[feat_layer])
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"epoch loss: {epoch_loss}")
    return enc_model, dec_model
