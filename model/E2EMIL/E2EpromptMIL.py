import torch
import torch.nn as nn
import os, glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils.file_tool import save_hdf5
from torchvision import transforms
from utils.get_feature_extractor import get_extractor
import argparse
import h5py

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='End-to-End MIL with prompt')
# data settings
parser.add_argument('--slide_dir', type=str, default='./')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--image_size', type=int, default=224)
# model settings
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model', type=str, default='ResNet50')
parser.add_argument('--custom_pretrained', type=str, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    print('%s Extractor'%args.model)
    model = get_extractor(args.model, custom_pretrained=args.custom_pretrained)
    model = model.to(device)