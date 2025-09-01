from __future__ import print_function

import argparse
import os
from datetime import datetime

import pandas as pd

from utils.data_utils import *
from utils.core_util import run_clam, run_transmil, run_abmil, run_dftdmil, run_frmil, define_model
from data.wsi_dataset_generic import Generic_MIL_Dataset
from utils import runner
import torch


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--repeat', type=int, default=5,
                    help='number of repeated experiments')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--test_data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--test_label', type=str, default='label',
                    help='test_label')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--n_classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--best_run', type=int, default=0)
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'mag'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['tiny', 'ultra_small', 'small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--ft_model', type=str, default='ResNet50',
                    choices=['ResNet50', 'ResNet50_prompt', 'ResNet50_deep_ft_prompt',
                             'ResNet50_simclr', 'ResNet50_simclr_prompt',
                             'ViT_S_16', 'ViT_S_16_prompt',
                             'ViT_S_16_dino', 'ViT_S_16_dino_prompt', 'ViT_S_16_dino_deep_ft_prompt',
                             'ViT_T_16', 'ViT_T_16_prompt', 'ViT_S_16_deep_ft_prompt', 'hipt', 'PLIP', 'UNI', 'CONCH'],)
parser.add_argument('--mil_method', type=str, default='CLAM_SB', help='mil method')
parser.add_argument('--task', type=str)
parser.add_argument('--accumulate_grad_batches', type=int, default=1,)
parser.add_argument('--use_h5', action='store_true', default=False, help='use h5 files')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
### DFTD specific options
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
### FRMIL specific options
parser.add_argument('--shift_feature', action='store_true', default=False, help='shift feature')
parser.add_argument('--drop_data', action='store_true', default=False, help='drop data')
parser.add_argument('--balanced_sample', action='store_true', default=False, help='balanced bag')
parser.add_argument('--n_heads', type=int, default=1, help='number of heads')
parser.add_argument('--mag', type=float, default=1.0, help='magnitude')
### DFP
parser.add_argument('--dfp', action='store_true', default=False)
parser.add_argument('--dfp_discrim', action='store_true', default=False)
parser.add_argument('--prompt_initialisation', type=str, default='gaussian', help='prompt init')
parser.add_argument('--prompt_aggregation', type=str, default='multiply', choices=['multiply', 'add', 'prepend'], help='prompt aggregation method')
parser.add_argument('--number_prompts', type=int, default=1)
parser.add_argument('--prompt_epoch', type=int, default=10)
### HIPT
parser.add_argument('--pretrain_4k',    type=str, default='None', help='Whether to initialize the 4K Transformer in HIPT', choices=['None', 'vit4k_xs_dino'])
parser.add_argument('--pretrain_WSI',    type=str, default='None')
parser.add_argument('--freeze_4k',      action='store_true', default=False, help='Whether to freeze the 4K Transformer in HIPT')
parser.add_argument('--freeze_WSI',     action='store_true', default=False, help='Whether to freeze the WSI Transformer in HIPT')
parser.add_argument('--top_k', type=int, default=0)
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def change_ft_pth(csv, old_ft='R50_features', new_ft='ViT_features'):
    for i in range(len(csv)):
        slide_ft_pth = csv.iloc[i, 0]
        if 'TT' not in slide_ft_pth:
            slide_ft_pth = slide_ft_pth.replace('processed_slide', 'processed_slide/source')
        if 'VIT_features' in slide_ft_pth:
            slide_ft_pth = slide_ft_pth.replace('VIT_features', new_ft)
        elif 'ResNet50' in slide_ft_pth:
            slide_ft_pth = slide_ft_pth.replace('ResNet50', new_ft)
        else:
            slide_ft_pth = slide_ft_pth.replace(old_ft, new_ft)
        csv.iloc[i, 0] = slide_ft_pth

    return csv

def main(args):
    # create results directory if necessary
    # args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code))
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    print('Results will be saved @ ', args.results_dir)
    logger = Logger(args.results_dir)
    logger(f"Save dir: {args.results_dir}")
    label_dict = {'WHO_label':{0:0,1:1,2:2},
                   'IDH_label':{0:0,1:1}} #0: 'HP', 1: 'NORM',2: 'TA.HG', 3: 'TA.LG',4: 'TVA.HG',5: 'TVA.LG'
    # data_root = os.environ.get('PBS_JOBFS') if args.task == 'IDH' else '/scratch/iq24/cc0395/WSI_Image'
    dataset = Generic_MIL_Dataset(csv_path='data/tumor_vs_normal_dummy_clean.csv',
                                  # data_root+'/IDH/train_filtered.csv',#'/scratch/sz65/cc0395/MedData/CAMELYON16/combine.csv',#'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                                  data_dir=args.data_root_dir,  # 'tumor_vs_normal_resnet_features'
                                  shuffle=False,
                                  print_info=False,
                                  seed=args.seed,
                                  label_dict=label_dict[args.test_label],
                                  patient_strat=False,
                                  dataset_name=args.task,
                                  use_h5=args.use_h5)
    # if 'hipt' in args.ft_model:
    #     csv_root = 'VIT_DINO_HIPT_features'
    # elif 'ViT' in args.ft_model:
    #     csv_root = 'ViT_features'
    # elif 'ResNet' in args.ft_model:
    #     csv_root = 'R50_features'
    # elif 'PLIP' in args.ft_model:
    #     csv_root = 'PLIP_features'
    # elif 'UNI' in args.ft_model:
    #     csv_root = 'UNI_features'
    # elif 'CONCH' in args.ft_model:
    #     csv_root = 'CONCH_features'
    # else:
    #     raise NotImplementedError

    csv_root = 'SN_R50_features'
    mean_fpr = np.linspace(0, 1, 100)               
    accs, f1s, aucs, mean_tpr = [], [], [], []
    # we are conducting 5 fold cross validation
    for n_fold in range(5):
        train_csv = f'{args.data_root_dir}/folds/fold_{n_fold+1}_train.csv'
        valid_csv = f'{args.data_root_dir}/folds/fold_{n_fold+1}_val.csv'
        test_csv = f'{args.data_root_dir}/folds/fold_{n_fold+1}_test.csv'
        train_csv = change_ft_pth(pd.read_csv(train_csv), old_ft='R50_features', new_ft=csv_root)
        valid_csv = change_ft_pth(pd.read_csv(valid_csv), old_ft='R50_features', new_ft=csv_root)
        test_csv = change_ft_pth(pd.read_csv(test_csv), old_ft='R50_features', new_ft=csv_root)

        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                        csv_path={
                                                                                'train': train_csv,
                                                                                'valid': valid_csv,
                                                                                'test': test_csv,},
                                                                        preprocess=True if args.task not in ['CAM16', 'BRIGHT'] else False,
                                                                        num_class=args.n_classes,
                                                                        test_label=args.test_label,
                                                                        balance=args.balanced_sample,
                                                                        top_k=args.top_k)

        datasets = (train_dataset, val_dataset, test_dataset)
        print(' '.join(f'--{k}={v} \n' for k, v in vars(args).items()))
        if args.testing:
            metarunner = runner.Meta(args, logger)
            run_acc, run_f1, run_auc, run_fpr, run_tpr = metarunner(n_fold, datasets)
            accs.append(run_acc)
            f1s.append(run_f1)
            aucs.append(run_auc)
            # Interpolate TPR at common FPR points
            interp_tpr = np.interp(mean_fpr, run_fpr, run_tpr)
            interp_tpr[0] = 0.0
            mean_tpr.append(interp_tpr)
        else:
            seed = int(datetime.now().timestamp())
            seed_torch(seed)
            metarunner = runner.Meta(args, logger)
            run_acc, run_f1, run_auc, run_fpr, run_tpr = metarunner(n_fold, datasets)
            accs.append(run_acc)
            f1s.append(run_f1)
            aucs.append(run_auc)
    print(f'Accuracies: avg: {np.mean(accs):.4f} std: {np.std(accs):.4f} best: {np.max(accs):.4f}')
    print(f'F1s: avg: {np.mean(f1s):.4f} std: {np.std(f1s):.4f} best: {np.max(f1s):.4f}')
    print(f'AUCs: avg: {np.mean(aucs):.4f} std: {np.std(aucs):.4f} best: {np.max(aucs):.4f}')
    print(f'Best run based on AUC: {np.argmax(aucs)}')
    # Calculate mean and standard deviation
    mean_tpr = np.mean(mean_tpr, axis=0)
    mean_tpr[-1] = 1.0
    dataset_stat = {
                'mean_fpr': mean_fpr,
                'mean_tpr': mean_tpr,
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
            }
    # save the dataset_stat
    np.save(os.path.join(args.results_dir, 'dataset_stat.npy'), dataset_stat)
    

if __name__ == '__main__':
    main(args)