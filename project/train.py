import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models
import numpy as np
from models import SRCNN, loss_net
import os
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import Resize, ToTensor

def upsample(img, factor):
    w, h = img.size
    return img.resize((int(w*factor), int(h*factor)))

def downsample(img, factor=4.0):
    return upsample(img, 1./factor)

def train(train_params, model_params, args, train_loader, test_loader):
    super_resolver = SRCNN().to(args.device)
    feat = loss_net().to(args.device)
    feat_layer = model_params['feat_layer']

    optimizer = Adam(super_resolver.parameters(), lr = train_params['lr'])
    mse_loss = nn.MSELoss()
    transform_batch = transforms.Compose([downsample, ToTensor()])


    for epoch in range(train_params['epochs']):
        super_resolver.train()
        epoch_loss = 0

        for batch_num, (sample_x, sample_y) in enumerate(train_loader):

            optimizer.zero_grad()

            pred = super_resolver(sample_x)
            f_hat = feat(pred)[feat_layer]
            f_gold = feat(sample_y)[feat_layer]

            C_j = f_gold.shape[0]
            H_j = f_gold.shape[1]
            W_j = f_gold.shape[2]
            loss = 1/(C_j*H_j*W_j) * mse_loss(f_hat, f_gold)
            loss = loss.to(args.device)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"epoch loss: {epoch_loss}")
    return super_resolver
