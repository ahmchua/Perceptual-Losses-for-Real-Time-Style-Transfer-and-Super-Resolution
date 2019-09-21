import argparse
import os
import torch
import sys

from train import *
from data import *
from models import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", default=os.path.join("..", "data", "train"), help="The relative path to the validation data directory")
    parser.add_argument("--test_data_dir", default=os.path.join("..", "data", "test"), help="The relative path to the validation data directory")
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
