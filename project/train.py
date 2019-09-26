import torch
import os
import logging
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from models import *

def upsample(img, factor):
    w, h = img.size
    return img.resize((int(w*factor), int(h*factor)))

def downsample(img, factor=4.0):
    return upsample(img, 1./factor)

def train(train_params, model_params, args, train_loader, test_loader):
    #if args.model_type == "srcnn":
    super_resolver = SRCNN().to(args.device)
    #elif args.model_type == "srres":
    #    super_resolver = SRResnet().to(args.device)
    #elif args.model_type == "srres2":
    #    super_resolver = SRResnet2().to(args.device)
    feat = loss_net().to(args.device)
    feat_layer = model_params['feat_layer']
    l1loss = nn.L1Loss()

    optimizer = Adam(super_resolver.parameters(), lr = train_params['lr'])
    mse_loss = nn.MSELoss()
    transform_batch = transforms.Compose([downsample, ToTensor()])
    path = "./model_checkpoints/"
    try:
      os.mkdir(path)
    except:
      pass
    for epoch in range(train_params['epochs']):
        super_resolver.train()
        epoch_loss = 0

        for batch_num, (sample_x, sample_y) in enumerate(train_loader):
            #if batch_num%25 == 0:
              #print(f"batch_num: {batch_num}")
            optimizer.zero_grad()

            if torch.cuda.is_available():
                sample_x = sample_x.to('cuda')
                sample_y = sample_y.to('cuda')

            pred = super_resolver(sample_x)
            f_hat = feat(pred)
            f_gold = feat(sample_y)

            l1_loss = l1loss(pred, sample_y)

            # Extract loss from a single layer
            if args.percep_loss == "l2_single":
                pred = f_hat[feat_layer]
                gold = f_gold[feat_layer]
                C_j = gold.shape[1]
                H_j = gold.shape[2]
                W_j = gold.shape[3]
                percep_loss = 1/(C_j*H_j*W_j)*torch.dist(pred, gold, p=2)
            elif args.percep_loss == "mse_single":
                pred = f_hat[feat_layer]
                gold = f_gold[feat_layer]
                percep_loss = mse_loss(pred, gold)
            elif args.percep_loss == "l2_multi":
                percep_loss = 0.0
                for name in f_hat:
                    pred = f_hat[name]
                    gold = f_gold[name]
                    C_j = gold.shape[1]
                    H_j = gold.shape[2]
                    W_j = gold.shape[3]
                    percep_loss += 1/(C_j*H_j*W_j) * torch.dist(pred, gold, p=2)
            elif args.percep_loss == "mse_multi":
                percep_loss = 0.0
                for name in f_hat:
                    pred = f_hat[name]
                    gold = f_gold[name]
                    percep_loss += mse_loss(pred, gold)

            print(f"percep loss: {percep_loss}")
            print(f"l1 loss: {l1_loss}")
            loss = train_params['percep_weight']*percep_loss + train_params['l1_weight']*l1_loss

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: {epoch_loss}")

        if epoch % 10 == 0:
            torch.save(super_resolver.state_dict(), path + args.model_name+f"_{epoch}.pth")

    return super_resolver
