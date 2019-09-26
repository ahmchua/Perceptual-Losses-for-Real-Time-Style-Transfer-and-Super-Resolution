import argparse
import os
import torch
import sys
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import DataLoader
from train import train
from data import MyCoco
from models import *

def upsample(img, factor):
    w, h = img.size
    return img.resize((int(w*factor), int(h*factor)))

def downsample(img, factor=4.0):
    return upsample(img, 1./factor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", default=os.path.join("..", "data", "val2017"), help="The relative path to the validation data directory")
    parser.add_argument("--train_annotation", default=os.path.join("..", "data", "annotations", "instances_val2017.json"), help="The relative path to the validation data directory")
    parser.add_argument("--epochs", type=int, default=51, help="Number of epochs to train")
    parser.add_argument("--noise_factor", type=float, default=0.0, help="Number of epochs to train")
    parser.add_argument("--percep_weight", type=float, default=1.0, help="Weighting of Perceptual Loss")
    parser.add_argument("--l1_weight", type=float, default=1.0, help="Weighting of L1 Loss")
    parser.add_argument("--model_name", type=str, default="srcnn", help="Weighting of L1 Loss")
    parser.add_argument("--percep_loss", type=str, default="l2", help="Type of perceptual loss")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning Rate")



    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    train_params = {'epochs': args.epochs, 'lr':args.lr, 'batch_size': 8, 'pin_memory':True, 'noise_factor':args.noise_factor, 'percep_weight':args.percep_weight, 'l1_weight':args.l1_weight}
    model_params = {'feat_layer':'relu2_2'}

    print(f"ARGUMENTS: {args}\n")
    print(f"TRAIN PARAMS: {train_params}\n")
    print(f"MODEL_PARAMS: {model_params}\n")

    target_transform = transforms.Compose([Resize((256,256)), ToTensor()])
    input_transform = transforms.Compose([Resize((256,256)), downsample, ToTensor()])

    train_dataset_og = MyCoco(
        root = args.train_images,
        annFile = args.train_annotation,
        noise_factor=train_params['noise_factor'],
        input_transform=input_transform,
        target_transform=target_transform
    )

    lengths = (int(0.1*len(train_dataset_og)), int(0.9*len(train_dataset_og)))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_og, lengths)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_params['batch_size'], pin_memory=train_params['pin_memory'])

    super_resolver = train(train_params, model_params, args, train_loader, test_loader=None)
