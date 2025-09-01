import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support
from sklearn.metrics import auc as calc_auc
from copy import deepcopy

from tqdm import tqdm
from sklearn.metrics import auc as calc_auc
import torch.nn.functional as F
from model.CLAM.model_clam import CLAM_MB, CLAM_SB
from model.ABMIL.model_abmil import Attention, GatedAttention
from model.CLAM.model_r2t_clam import R2T_CLAM
from model.ABMIL.model_r2t_abmil import R2T_ABMIL
from model.TransMIL.model_transmil import TransMIL
from model.FRMIL.model_frmil import FRMIL
from model.R2TMIL.model_r2tmil import RRTMIL
from model.prompter import prompt_init
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import CategoriesSampler
from utils.core_util import define_model, RAdam, get_split_loader, FeatMag
from utils.trainer_util import calculate_error, Meter, compute_accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####Temporarily added#####
def define_clam(args):
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.mil_method != 'mil':
        if 'ResNet50' in args.ft_model or 'UNI' in args.ft_model:
            model_size = 'small'
        elif 'PLIP' in args.ft_model or 'CONCH' in args.ft_model:
            model_size = 'medium'
        elif 'ViT_S_16' in args.ft_model:
            model_size = 'ultra_small'
        elif 'ViT_T_16' in args.ft_model:
            model_size = 'tiny'
        else:
            raise NotImplementedError
        model_dict.update({"size_arg": model_size})

    if args.subtyping:
        model_dict.update({'subtyping': True})

    if args.B > 0:
        model_dict.update({'k_sample': args.B})

    if args.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes=2)
        instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()

    if args.mil_method == 'CLAM_SB':
        model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    elif args.mil_method == 'CLAM_MB':
        model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
    else:
        raise NotImplementedError
    model.relocate()
    return model

def define_transmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError

    model = TransMIL(model_size, n_classes=args.n_classes)
    model = model.to(device)
    return model

def define_abmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    if args.mil_method == 'abmil_att':
        model = Attention(model_size, args.n_classes)
    elif args.mil_method == 'abmil_gatedatt':
        model = GatedAttention(model_size, args.n_classes)
    else:
        raise NotImplementedError
    model = model.to(device)
    return model

def define_frmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    size_dict = {"tiny": [192, 128, 128], "ultra_small": [384, 192, 128],
                      "small": [1024, 512, 256], "big": [2048, 512, 384]}

    model = FRMIL(args, size_dict[model_size])
    model = model.to(device)
    return model

def define_r2tmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    size_dict = {"tiny": [192], "ultra_small": [384],
                      "small": [1024], "big": [2048]}
    model_params = {
        'input_dim': size_dict[model_size][0],
        'n_classes': args.n_classes,
        'dropout': 0.1,
        'act': 'relu',
        'region_num': 4,
        'pos': None,
        'pos_pos': 0,
        'pool': 'attn',
        'peg_k': 7,
        'drop_path': 0,
        'n_layers': 2,
        'n_heads': 8,
        'attn': 'rmsa',
        'da_act': 'tanh',
        'trans_dropout': 0.1,
        'ffn': False,
        'mlp_ratio': 4,
        'trans_dim': 64,
        'epeg': True,
        'min_region_num': 0,
        'qkv_bias': True,
        'epeg_k': 15,
        'epeg_2d': False,
        'epeg_bias': True,
        'epeg_type': 'attn',
        'region_attn': 'native',
        'peg_1d': False,
        'cr_msa': True,
        'crmsa_k': 3,
        'all_shortcut': False,
        'crmsa_mlp': False,
        'crmsa_heads': 8,
    }
    model = RRTMIL(**model_params).to(device)
    return model


def define_model(args):
    if args.mil_method == 'r2tmil_abmil':
        model = R2T_ABMIL(args)
    elif args.mil_method == 'r2tmil_CLAM':
        model = R2T_CLAM(args)
    elif 'CLAM' in args.mil_method:
        model = define_clam(args)
    elif 'transmil' in args.mil_method:
        model = define_transmil(args)
    elif 'abmil' in args.mil_method:
        model = define_abmil(args)
    elif 'frmil' in args.mil_method:
        model = define_frmil(args)
    else:
        raise NotImplementedError
    return model

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

class Prompter(nn.Module):
    def __init__(self, args):
        super(Prompter, self).__init__()
        self.prompt_embeddings, self.prompt_norm = prompt_init(args)
        self.prompt_aggregation = args['prompt_aggregation']
        # self.gate = nn.Linear(args['prompt_size'] * 2, args['prompt_size']).to(self.prompt_embeddings.device)
        self.prompt_dropout = nn.Dropout(args['prompt_dropout'])
    def learnable_parameters(self):
        learnable_params = [self.prompt_embeddings]
        # learnable_params += list(self.gate.parameters())
        learnable_params    += list(self.prompt_dropout.parameters())
        return learnable_params

class Meta(nn.Module):
    """Diversity-Aware Prompt Initilization Based on Fast Meta Learning.
    """
    def __init__(self, args, logger=None):
        super(Meta, self).__init__()
        self.args = args
        self.model = define_model(args)
        dfp_dict = {'init': args.prompt_initialisation,
                    'number_prompts': args.number_prompts,
                    'prompt_aggregation': args.prompt_aggregation,
                    'prompt_size': self.model.size[0],
                    'prompt_dropout': args.prompt_dropout}
        self.prompter = Prompter(dfp_dict)
        self.logger = logger
        # print number of trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger('Number of trainable parameters in MIL: %d' % num_params)
        num_params_prompt = sum(p.numel() for p in self.prompter.parameters() if p.requires_grad)
        self.logger('Number of trainable parameters in Prompter: %d' % num_params_prompt)

    def summary_frmil(self, loader, args=None, set='val', thrs=0.5):
        self.model.eval()
        loss_meter = Meter()
        acc_meter = Meter()

        tqdm_gen = tqdm(loader)

        with torch.no_grad():
            for batch_idx, (data_ft, stain_ft, label) in enumerate(tqdm_gen, 1):
                data_ft, stain_ft, label = data_ft.to(device), stain_ft.to(device), label.to(device)
                prompted_data = self.get_prompted_ft_based_on_stain(data_ft, stain_ft, self.prompter_gather)
                logits = self.model(prompted_data)
                loss = F.cross_entropy(logits, label.long())

                acc = compute_accuracy(logits, label.long())
                logits = F.softmax(logits, dim=1)[:, label.long()]
                loss_meter.update(loss.item())
                acc_meter.update(acc)
                acc_meter.update_gt(label.data.cpu().numpy()[0], logits.data.squeeze().cpu().numpy())

        if set == 'val':
            acc, auc, fscore, op_thrs = acc_meter.acc_auc()
            return loss_meter.avg(), acc, auc, fscore, op_thrs
        else:
            acc, auc, fscore, op_thrs = acc_meter.acc_auc(thrs)
            return loss_meter.avg(), acc, auc, fscore

    def summary(self, loader, n_classes, modelname='clam'):
        self.model.eval()
        test_error = 0.

        all_probs = np.zeros((len(loader), n_classes))
        all_labels = np.zeros(len(loader))
        all_preds = np.zeros(len(loader))

        slide_ids = loader.dataset.slide_data['File Path']
        slide_pred = {'slide_id': [], 'pred': [], 'gt': [], 'error': []}
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(loader)):
                data_ft, label = data
                data_ft, label = data_ft.to(device), label.to(device)
                if self.args.global_prompt:
                    prompt = self.prompter.prompt_embeddings.squeeze(0) + torch.mean(data_ft, dim=1, keepdim=True)
                else:
                    prompt = self.prompter.prompt_embeddings.squeeze(0)
                prompted_data = torch.cat([data_ft, prompt], dim=0)
                if 'CLAM' in modelname:
                    logits, Y_prob, Y_hat, _, results_dict = self.model(prompted_data, label, instance_eval=True,
                                                                   return_features=True)
                elif modelname == 'transmil':
                    prompted_data = prompted_data.unsqueeze(0)
                    results_dict = self.model(data=prompted_data, label=label)
                    Y_prob = results_dict['Y_prob']
                    Y_hat = results_dict['Y_hat']
                elif 'abmil' in modelname:
                    logits, Y_prob, Y_hat, _ = self.model.forward(prompted_data)
                elif 'frmil' in modelname:
                    logits = self.model(prompted_data)
                    Y_prob = F.softmax(logits, dim=1)
                    Y_hat = torch.topk(logits, 1, dim=1)[1]
                elif 'r2tmil' in modelname:
                    logits = self.model(data_ft)
                    Y_prob = F.softmax(logits, dim=1)
                    Y_hat = torch.topk(logits, 1, dim=1)[1]
                else:
                    raise NotImplementedError

                probs = Y_prob.cpu().numpy()
                all_probs[batch_idx] = probs
                all_labels[batch_idx] = label.item()
                all_preds[batch_idx] = Y_hat.item()

                error = calculate_error(Y_hat, label)
                slide_pred['slide_id'].append(slide_ids[batch_idx])
                slide_pred['pred'].append(Y_hat.item())
                slide_pred['gt'].append(label.item())
                slide_pred['error'].append(error)
                test_error += error

        test_error /= len(loader)
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))
        f1 = f1_score(all_labels, all_preds, average='macro')
        #out slide_pred to csv
        slide_pred = pd.DataFrame(slide_pred)
        slide_pred.to_csv(os.path.join(self.args.results_dir, 'slide_pred.csv'), index=False)
        return test_error, auc, f1

    def stain_clustering(self):
        if not self.args.use_proto:
            print('=============Not using stain prototype=============')
            self.num_coarse_stain_classes = 1
        else:
            print('=============Loading stain prototype=============')
            # stain_proto_path = f'./data/pre_extracted_color_feature/{self.args.task}'
            # self.stain_prototype = torch.load('%s/Train/prototype.pt'%stain_proto_path).to(device)
            # self.num_coarse_stain_classes = self.stain_prototype.size(0)
            self.num_coarse_stain_classes = self.args.number_panther_proto
    def get_prompted_ft_based_on_stain(self, h, h_stain, prompter_gather=None):
        prompted_image = []
        # if os.path.exists(f'{self.args.data_root_dir}/'):
        reform = False
        if len(h.size()) > 2:
            b, n, _ = h.size()
            for i in range(h.size(0)):
                prompted_image_batch = []
                h_i = h[i]
                h_stain_i = h_stain[i]
                unique_idx = torch.unique(h_stain_i).to(torch.long)
                for i in unique_idx:
                    idx_h = torch.where(h_stain_i == i)[0].to(torch.long)
                    prompted_image_batch.append(
                        prompter_gather[i](h_i[idx_h])
                    )
                prompted_image.append(torch.cat(prompted_image_batch, dim=0))
            prompted_image = torch.cat(prompted_image, dim=0)
            prompted_image = prompted_image.view(b, n, -1)
        else:
            indices = h_stain
            unique_idx = torch.unique(indices).to(torch.long)
            for i in unique_idx:
                idx_h = torch.where(indices == i)[0].to(torch.long)
                prompted_image.append(
                    prompter_gather[0](h[idx_h])
                )
            prompted_image = torch.cat(prompted_image, dim=0)
        return prompted_image

    def get_prompted_ft(self, h, prompter_gather=None):
        reform = False
        if len(h.size()) > 2:
            reform = True
            b, n, _ = h.size()
            h = h.view(-1, h.size(-1))
        h = prompter_gather[0](h)
        if reform:
            h = h.view(b, n, -1)
        return h

    def get_optim(self):
        if self.num_coarse_stain_classes > 1:
            # cluster_counts = torch.tensor([50446, 99181, 85842, 154728, 106238]) #R50
            cluster_counts = torch.tensor([79414, 122713, 145406, 127209, 21693]) # PLIP
            cluster_weights = 1.0 / cluster_counts.float()
            cluster_weights = cluster_weights / cluster_weights.sum()
        else:
            cluster_weights = [1.0]
        self.prompter_gather, self.prompter_params_gather = [], []
        for i in range(self.num_coarse_stain_classes):
            self.prompter_gather.append(
                deepcopy(self.prompter)
            )
            for param in self.prompter_gather[i].parameters():
                param.requires_grad = True
            self.prompter_params_gather.append(
                {'params': self.prompter_gather[i].learnable_parameters(),
                 'lr':self.args.prompt_lr,
                 'weight_decay':self.args.prompt_reg,
                 'weight': cluster_weights[i]}
            )
        self.prompter_params_gather.append(
            {'params': filter(lambda p: p.requires_grad, self.model.parameters()),
             'lr': self.args.lr,
             'weight_decay': self.args.reg,
             'weight': 1.0}
        )
        print(f"============={len(self.prompter_gather)} Stain prototype loaded=============")
        if self.args.opt == "adam":
            optimizer = optim.Adam(self.prompter_params_gather)
        elif self.args.opt == 'adamw':
            optimizer = optim.AdamW(self.prompter_params_gather)
        elif self.args.opt == 'sgd':
            for i in range(len(self.prompter_params_gather)):
                self.prompter_params_gather[i]['momentum'] = 0.9
            optimizer = optim.SGD(self.prompter_params_gather)
        elif self.args.opt == 'radam':
            optimizer = RAdam(self.prompter_params_gather)
        else:
            raise NotImplementedError
        return optimizer

    def get_loss(self):
        if self.args.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes=self.args.n_classes)
            loss_fn = loss_fn.cuda()
        elif self.args.bag_loss == 'mag':
            loss_fn = FeatMag(margin=self.args.mag).cuda()
        else:
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def clam_runner(self, prompted_data, label, loss_fn):
        logits, Y_prob, Y_hat, _, instance_dict = self.model(prompted_data, label=label, instance_eval=True)
        loss = loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = self.args.bag_weight * loss + (1 - self.args.bag_weight) * instance_loss
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def transmil_runner(self, prompted_data, label, loss_fn):
        prompted_data = prompted_data.unsqueeze(0)
        results_dict = self.model(data=prompted_data, label=label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        total_loss = loss_fn(logits, label)
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def abmil_runner(self, prompted_data, label, loss_fn):
        logits, Y_prob, Y_hat, A = self.model.forward(prompted_data)
        total_loss = loss_fn(logits, label)
        error = 1. - Y_hat.eq(label).cpu().float().mean().item()
        return total_loss, error

    def frmil_runner(self, prompted_data, label, loss_fn, bce_weight, ce_weight):
        norm_idx = torch.where(label.cpu() == 0)[0].numpy()[0]
        ano_idx = 1 - norm_idx
        if self.args.drop_data:
            prompted_data = F.dropout(prompted_data, p=0.20)
        logits, query, max_c = self.model(prompted_data)

        # all losses
        max_c = torch.max(max_c, 1)[0]
        loss_max = F.binary_cross_entropy(max_c, label.float(), weight=bce_weight)
        loss_bag = F.cross_entropy(logits, label, weight=ce_weight)
        loss_ft = loss_fn(query[ano_idx, :, :].unsqueeze(0), query[norm_idx, :, :].unsqueeze(0),
                           w_scale=query.shape[1])
        loss = (loss_bag + loss_ft + loss_max) * (1. / 3)
        acc = compute_accuracy(logits, label)

        return loss, 1 - acc/100

    def r2tmil_runner(self, prompted_data, label):
        train_logits = self.model(prompted_data)
        loss = nn.CrossEntropyLoss()(train_logits.view(1,-1),label)
        acc = compute_accuracy(train_logits, label)
        return loss, 1 - acc / 100

    def forward(self, iter, datasets):
        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        if len(test_split) == 0:
            test_split = val_split
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        print('\nInit optimizer ...', end=' ')
        optimizer = self.get_optim()
        print('Done!')

        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        print('\nInit Loaders...', end=' ')
        if 'frmil' in self.args.mil_method:
            train_sampler = CategoriesSampler(train_split.labels,
                                              n_batch=len(train_split.slide_data),
                                              n_cls=self.args.n_classes,
                                              n_per=1)
            train_loader = DataLoader(dataset=train_split, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
            val_loader = DataLoader(dataset=val_split, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
            test_loader = DataLoader(dataset=test_split, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
        else:
            train_loader = get_split_loader(train_split, training=True, weighted=self.args.weighted_sample)
            val_loader = get_split_loader(val_split)
            test_loader = get_split_loader(test_split)
        print('Done!')

        scheduler = cosine_lr(
            optimizer,
            self.args.prompt_lr,
            len(train_loader)*self.args.max_epochs//5,
            len(train_loader)*self.args.max_epochs
        )

        if self.args.testing:
            model_ckp_pth = os.path.join(self.args.results_dir, 'best_model_%d.pt' % iter)
            print('**** Loading model checkpoint from %s ****' % model_ckp_pth)
            ckp = torch.load(model_ckp_pth)
            self.model.load_state_dict(ckp)

            prompt_ckp_pth = os.path.join(self.args.results_dir, 'best_prompt_%d.pt' % iter)
            print('**** Loading prompt checkpoint from %s ****' % prompt_ckp_pth)
            prompt_ckp = torch.load(prompt_ckp_pth)
            self.prompter_gather= prompt_ckp
            # if 'frmil' in self.args.mil_method:
            #     _, test_acc, test_auc, test_f1, test_thrs = self.summary_frmil(test_loader)
            # else:
            test_error, test_auc, test_f1 = self.summary(test_loader,
                                                       self.args.n_classes,
                                                       modelname=self.args.mil_method)
            test_acc = 1 - test_error
            self.logger('======> Test acc: {:.4f}, ROC AUC: {:.4f} Test F1: {:.4f}'.format(test_acc, test_auc, test_f1))
            return

        best_val_acc = 0
        best_val_f1 = 0
        best_val_auc = 0
        patience = 0
        max_patience = 20  # Number of epochs to wait before early stopping
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        best_prompt_save_pth = os.path.join(self.args.results_dir, "best_prompt_%d.pt" % iter)
        print('Training start')
        for epoch in range(self.args.max_epochs):
            initial_prompt = deepcopy(self.prompter_gather)
            torch.cuda.empty_cache()
            self.model.train()

            train_loss = 0.
            train_error = 0.

            if 'frmil' in self.args.mil_method:
                ce_weight = [i for i in train_loader.dataset.count_dict.values()]
                ce_weight = 1. / torch.tensor(ce_weight, dtype=torch.float)
                ce_weight = ce_weight.cuda()
                bce_weight = train_loader.dataset.pos_weight.cuda()
            tqdm_gen = tqdm(train_loader)
            for batch_idx, data in enumerate(tqdm_gen):
                data_ft, label = data
                data_ft, label = data_ft.to(device), label.to(device)
                if self.args.global_prompt:
                    prompt = self.prompter.prompt_embeddings.squeeze(0) + torch.mean(data_ft, dim=1, keepdim=True) * self.args.global_weight
                else:
                    prompt = self.prompter.prompt_embeddings.squeeze(0)
                prompted_data = torch.cat([data_ft, prompt], dim=0)

                if 'CLAM' in self.args.mil_method:
                    total_loss, error = self.clam_runner(prompted_data, label, loss_fn)
                elif 'transmil' in self.args.mil_method:
                    total_loss, error = self.transmil_runner(prompted_data, label, loss_fn)
                elif 'abmil' in self.args.mil_method:
                    total_loss, error = self.abmil_runner(prompted_data, label, loss_fn)
                elif 'frmil' in self.args.mil_method:
                    total_loss, error = self.frmil_runner(prompted_data, label, loss_fn, bce_weight, ce_weight)
                elif 'r2tmil' in self.args.mil_method:
                    total_loss, error = self.r2tmil_runner(data_ft, label)
                else:
                    raise NotImplementedError
                train_error += error
                total_loss.backward()
                loss_value = total_loss.item()
                train_loss += loss_value
                # step
                if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
                    # Scale gradients manually
                    for group in self.prompter_params_gather:
                        weight = group["weight"]
                        for param in group["params"]:
                            if param.grad is not None:
                                param.grad.data *= weight  # Scale gradients by the specified weight
                        # print(f'Optimizing Prompter+MIL {weight}...')
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
            # calculate loss and error for epoch
            train_loss /= len(train_loader)
            train_error /= len(train_loader)

            if epoch % 1 == 0:
                val_error, val_auc, val_f1 = self.summary(val_loader,
                                                            self.args.n_classes,
                                                            modelname=self.args.mil_method)
                val_acc = 1 - val_error
                log_info = '======> @ Epoch {} Val acc: {:.4f}, ROC AUC: {:.4f}, Val F1: {:.4f}, LR: {:.4f}'.format(epoch,
                                                                                                                val_acc,
                                                                                                                val_auc,
                                                                                                                val_f1,
                                                                                                                optimizer.param_groups[0]['lr'])

                self.logger(log_info)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_prompt = deepcopy(self.prompter_gather)
                    torch.save(self.model.state_dict(), best_model_save_pth)
                    torch.save(best_prompt, best_prompt_save_pth)
                    self.logger('******Best Model saved @ %s*******' % best_model_save_pth)
                    patience = 0  # Reset patience when performance improves
                else:
                    patience += 1
                    self.logger(f'Patience: {patience}/{max_patience}')
                if patience >= max_patience:
                    self.logger(f'Early stopping triggered after {epoch} epochs')
                    break
        
        # final testing
        print('**** Loading model checkpoint from %s ****' % best_model_save_pth)
        ckp = torch.load(best_model_save_pth)
        self.model.load_state_dict(ckp)

        print('**** Loading prompt checkpoint from %s ****' % best_prompt_save_pth)
        prompt_ckp = torch.load(best_prompt_save_pth)
        self.prompter_gather= prompt_ckp

        test_error, test_auc, test_f1 = self.summary(test_loader,
                                                    self.args.n_classes,
                                                    modelname=self.args.mil_method)
        self.logger('========> Test Acc: %.3f F1: %.3f AUC: %.3f' % (1-test_error, test_f1, test_auc))
        self.logger('=============================== [Iter %d] Done ===============================' % (iter))
        return test_error, test_f1, test_auc
    