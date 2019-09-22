import argparse
import os
import torch
import sys
import torchvision.datasets as dataset

from train import train
from data import MyCoco
from models import *
from val_grader.tests import downsample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", default=os.path.join("..", "data", "train2017"), help="The relative path to the validation data directory")
    parser.add_argument("--train_annotation", default=os.path.join("..", "data", "annotations", "instances_train2017.json"), help="The relative path to the validation data directory")
    parser.add_argument("--test_images", default=os.path.join("..", "data", "val2017"), help="The relative path to the validation data directory")
    parser.add_argument("--batch_size", default= 16, type=int, help="The batch size for training")
    parser.add_argument("--model_params", type=str, default=os.path.join("params", "model_conv.json"), help="Path to json file with model parameters")
    parser.add_argument("--train_params", type=str, default=os.path.join("params", "model_conv.json"), help="Path to json file with train parameters")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    with open(args.model_params, "r") as f:
        model_params = json.load(f)
    with open(args.train_params, "r") as f:
        train_params = json.load(f)

    train_loader = MyCoco(root = train_path, annFile = train_annotation_path, transforms = )
    test_loader = MyCoco(root = test_path, annFile = test_annotation_path, transforms = )

    train(train_params, model_params, train_loader, test_loader)
