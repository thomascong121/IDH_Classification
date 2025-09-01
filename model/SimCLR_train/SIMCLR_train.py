import sys
sys.path.insert(1, '/scratch/sz65/cc0395/WSI_prompt')
from model.SimCLR_train.simclr import SimCLR
import yaml
import torch,os
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse
from utils.file_tool import generate_csv

# def generate_csv(args):
#     all_slide_csv = glob.glob(args.csv_path + '/*_patch.csv')
#     print(all_slide_csv)
#     all_slide_content = {'patch_id':[]}
#     for slide_csv in all_slide_csv:
#         slide_content = pd.read_csv(slide_csv)
#         for patch_base in slide_content['patch_id']:
#             if 'train' in slide_csv:
#                 patch_root = os.path.join(args.csv_path, '_Train_All', patch_base)
#             elif 'test' in slide_csv:
#                 patch_root = os.path.join(args.csv_path, '_Test_All', patch_base)
#             else:
#                 raise NotImplementedError
#             all_slide_content['patch_id'].append(patch_root)
#     df = pd.DataFrame.from_dict(all_slide_content)
#     print(df.head(10))
#     df.to_csv(args.csv_path + '/all_patches.csv', index=False)
#     print('Done, output csv: ', args.csv_path + '/all_patches.csv')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--slide_dir', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--config_path', type=str, default=None)
parser.add_argument('--image_size', type=int, default=224)
args = parser.parse_args()
config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
csv_path = args.csv_path + '/all_patches.csv'
if not os.path.exists(csv_path):
    generate_csv(args)
gpu_ids = eval(config['gpu_ids'])
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
dataset = DataSetWrapper(csv_path, config['batch_size'], **config['dataset'])
simclr = SimCLR(dataset, config)
simclr.train()