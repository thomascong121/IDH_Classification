import scipy
# from timm.models.helpers import update_pretrained_cfg_and_kwargs, load_pretrained, load_custom_pretrained
# from timm.models.vision_transformer import _cfg, default_cfgs, checkpoint_filter_fn, build_model_with_cfg
from copy import deepcopy
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer, required
import time
import logging
import json
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import get_split_loader, CategoriesSampler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support
from utils.trainer_util import *
from model.CLAM.model_clam import CLAM_MB, CLAM_SB, CLAM_SB_prompted
from model.TransMIL.model_transmil import TransMIL
from model.ABMIL.model_abmil import Attention, GatedAttention
from model.DTFDMIL.Attention import Attention_Gated as DFTD_Attention
from model.DTFDMIL.Attention import Attention_with_Classifier
from model.DTFDMIL.network import Classifier_1fc, DimReduction
from model.FRMIL.model_frmil import FRMIL
from sklearn.decomposition import PCA

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def replace_bn_layers(model_to, model_from):
    # load model_from bn to model_to bn
    model_to_dict = model_to.state_dict()
    model_from_dict = model_from.state_dict()
    for k, v in model_to_dict.items():
        if 'bn' in k:
            model_to_dict[k] = model_from_dict[k]
    model_to.load_state_dict(model_to_dict)
    return model_to

def replace_ln_layers(model_to, model_from):
    # load model_from bn to model_to bn
    model_to_dict = model_to.state_dict()
    model_from_dict = model_from.state_dict()
    count = 0
    for k, v in model_to_dict.items():
        if 'norm' in k:
            model_to_dict[k] = model_from_dict[k]
            count += 1
            print(k)
    model_to.load_state_dict(model_to_dict)
    print('Number of layer norm layers replaced: ', count)
    return model_to

def load_pretrained_vit(model, model_name, variant, vit_kwargs, custom_pretrained, pretrained=True):
    if pretrained:
        print('==========> Load pretrained %s'%(model.__class__.__name__))
    pretrained_cfg = deepcopy(default_cfgs[variant])
    update_pretrained_cfg_and_kwargs(pretrained_cfg, vit_kwargs, None)
    pretrained_cfg.setdefault('architecture', variant)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat
    num_classes_pretrained = getattr(model, 'num_classes', vit_kwargs.get('num_classes', 1000))
    pretrained_custom_load = 'npz' in pretrained_cfg['url']
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=vit_kwargs.get('in_chans', 3),
                filter_fn=checkpoint_filter_fn,
                strict=False)
    if 'dino' in model_name:
        model = custom_load_pretrained(model, model_name, custom_pretrained)
    return model

def custom_load_pretrained(model, model_name, custom_pretrained):
    print('Custom load pretrained model from ', custom_pretrained)
    ckp = torch.load(custom_pretrained)
    from collections import OrderedDict
    pretrained_state_dict = OrderedDict()
    if 'dino' in model_name:
        print('Load DINO model from ', custom_pretrained)
        new_state_dict = model.state_dict()
        for k, v in ckp['teacher'].items():
            name_list = k.split('.')
            if name_list[0] == 'backbone':
                name = '.'.join(name_list[1:])
            else:
                name = k
            pretrained_state_dict[name] = v
        for k, v in model.state_dict().items():
            if k not in pretrained_state_dict:
                print(k)
                new_state_dict[k] = v
            else:
                new_state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(new_state_dict, strict=True)
    elif 'simclr' in model_name:
        print('Load SimCLR model from ', custom_pretrained)
        new_state_dict = model.state_dict()
        for k, v in ckp.items():
            name_list = k.split('.')
            if 'module' in k and 'features' in k:
                name = '.'.join(name_list[2:])
            elif 'module' in k:
                name = '.'.join(name_list[1:])
            else:
                name = k
            pretrained_state_dict[name] = v
        for k, v in model.state_dict().items():
            if k not in pretrained_state_dict:
                print('Not found in pretrained: ', k)
                new_state_dict[k] = v
            else:
                new_state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(new_state_dict, strict=True)
    return model

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0 and group['weight_decay'] is not None:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    elif args.opt == 'radam':
        optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def summary(model, loader, n_classes, modelname='clam', test_only=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    feature_to_label = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if 'clam' in modelname:
                logits, Y_prob, Y_hat, _, results_dict = model(data, label, instance_eval=True, return_features=True)
            elif modelname == 'transmil':
                data = data.unsqueeze(0)
                results_dict = model(data=data, label=label)
                Y_prob = results_dict['Y_prob']
                Y_hat = results_dict['Y_hat']
            elif 'abmil' in modelname:
                logits, Y_prob, Y_hat, _ = model.forward(data)
            elif 'frmil' in modelname:
                logits = model(data)
                Y_prob = F.softmax(logits, dim=1)
                Y_hat = torch.topk(logits, 1, dim=1)[1]
            else:
                raise NotImplementedError

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
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
    return patient_results, test_error, auc, f1
#
# class Accuracy_Logger(object):
#     """Accuracy logger"""
#     def __init__(self, n_classes):
#         super(Accuracy_Logger, self).__init__()
#         self.n_classes = n_classes
#         self.initialize()
#
#     def initialize(self):
#         self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
#
#     def log(self, Y_hat, Y):
#         Y_hat = int(Y_hat)
#         Y = int(Y)
#         self.data[Y]["count"] += 1
#         self.data[Y]["correct"] += (Y_hat == Y)
#
#     def log_batch(self, Y_hat, Y):
#         Y_hat = np.array(Y_hat).astype(int)
#         Y = np.array(Y).astype(int)
#         for label_class in np.unique(Y):
#             cls_mask = Y == label_class
#             self.data[label_class]["count"] += cls_mask.sum()
#             self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
#
#     def get_summary(self, c):
#         count = self.data[c]["count"]
#         correct = self.data[c]["correct"]
#
#         if count == 0:
#             acc = None
#         else:
#             acc = float(correct) / count
#
#         return acc, correct, count
#

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class FeatMag(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, feat_pos, feat_neg, w_scale=1.0):
        loss_act = self.margin - torch.norm(torch.mean(feat_pos, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_neg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um / w_scale

def define_clam(args):
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.mil_method != 'mil':
        if 'ResNet50' in args.ft_model:
            model_size = 'small'
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
        # if device.type == 'cuda':
        instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()

    # TODO: check if this is correct
    dfp_dict = {'init':args.prompt_initialisation,
                'number_prompts': args.number_prompts,
                'prompt_aggregation': args.prompt_aggregation,
                'prompt_disrim': args.dfp_discrim}
    if args.dfp:
        use_dfp = dfp_dict
    else:
        use_dfp = None
    if args.mil_method == 'CLAM_SB' and use_dfp is None:
        model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn, dfp_dict=use_dfp)
    elif args.mil_method == 'CLAM_SB' and use_dfp is not None:
        model = CLAM_SB_prompted(**model_dict, instance_loss_fn=instance_loss_fn, dfp_dict=use_dfp)
    elif args.mil_method == 'CLAM_MB':
        model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn, dfp_dict=use_dfp)
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
    dfp_dict = {'init':args.prompt_initialisation,
                'number_prompts': args.number_prompts,
                'prompt_aggregation': args.prompt_aggregation,
                'prompt_disrim': args.dfp_discrim}
    if args.dfp:
        use_dfp = dfp_dict
    else:
        use_dfp = None
    model = TransMIL(model_size, n_classes=args.n_classes, dfp_dict=use_dfp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    dfp_dict = {'init':args.prompt_initialisation,
                'number_prompts': args.number_prompts,
                'prompt_aggregation': args.prompt_aggregation,
                'prompt_disrim': args.dfp_discrim}
    if args.dfp:
        use_dfp = dfp_dict
    else:
        use_dfp = None
    if args.mil_method == 'abmil_att':
        model = Attention(model_size, args.n_classes, dfp_dict=use_dfp)
    elif args.mil_method == 'abmil_gatedatt':
        model = GatedAttention(model_size, args.n_classes, dfp_dict=use_dfp)
    else:
        raise NotImplementedError
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def define_dftdmil(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.drop_out:
        droprate = 0.25
    else:
        droprate = 0

    dimReduction = DimReduction(size_dict[model_size][0], size_dict[model_size][1], numLayer_Res=args.numLayer_Res).to(device)
    classifier = Classifier_1fc(size_dict[model_size][1], args.n_classes, droprate).to(device)
    attention = DFTD_Attention(size_dict[model_size][1]).to(device)
    attCls = Attention_with_Classifier(L=size_dict[model_size][1], num_cls=args.n_classes, droprate=droprate).to(device)
    return (dimReduction, classifier, attention, attCls)

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
    # TODO: check if this is correct
    dfp_dict = {'init':args.prompt_initialisation,
                'number_prompts': args.number_prompts,
                'prompt_aggregation': args.prompt_aggregation,
                'prompt_disrim': args.dfp_discrim}
    if args.dfp:
        use_dfp = dfp_dict
    else:
        use_dfp = None
    model = FRMIL(args, size_dict[model_size], dfp_dict=use_dfp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def define_model(args):
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'mag':
        loss_fn = FeatMag(margin=args.mag).cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()

    if 'CLAM' in args.mil_method:
        model = define_clam(args)
    elif args.mil_method == 'transmil':
        model = define_transmil(args)
    elif 'abmil' in args.mil_method:
        model = define_abmil(args)
    elif args.mil_method == 'dftdmil':
        model = define_dftdmil(args)
    elif args.mil_method == 'frmil':
        model = define_frmil(args)
    else:
        raise NotImplementedError

    return model, loss_fn

def get_ft_stat(model, test_image_dataset, type='prompt'):
    model.eval()
    test_ft_r50_dfp = []
    test_labels = []
    val_loader = get_split_loader(test_image_dataset)
    test_ft_stat = []
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    with torch.no_grad():
        for batch_idx, (imgs, label) in tqdm(enumerate(val_loader)):
            imgs, label = imgs.to(device), label.to(device)
            if type == 'prompt':
                ft = model(imgs, label, ft_only=True)
            elif type == 'ViT_base':
                ft = model(imgs, return_feature=True)
            elif type == 'prompt_2stage':
                logits, Y_prob, Y_hat, _, instance_dict = model(imgs, label=label, instance_eval=True)
                ft = instance_dict['prompted_features']
            else:
                ft = model(imgs, return_feature=True)
            label = label.repeat(ft.size(0))
            ft = pca.fit_transform(ft.cpu())
            test_ft_r50_dfp.append(ft)
            test_labels.append(label)
            test_ft_stat.append(np.mean(ft, axis=1))
    test_ft_r50_dfp = np.concatenate(test_ft_r50_dfp)
    test_labels = torch.cat(test_labels).cpu().numpy()
    test_ft_stat = np.concatenate(test_ft_stat)
    return test_ft_r50_dfp, test_labels, test_ft_stat

def run_clam(datasets, model, loss_fn, iter, args, logger):
    """
        train for a single fold
    """
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if len(test_split) == 0:
        test_split = val_split
        print('[Warning] Testing on valdation set')
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    if args.testing:
        ckp_pth = os.path.join(args.results_dir, 'best_%d.pt'%iter)
        print('**** Loading checkpoint from %s ****' % ckp_pth)
        ckp = torch.load(ckp_pth)

        for k, v in ckp.items():
            print(k)
        print('='*10)
        for k, v in model.state_dict().items():
            print(k)

        model.load_state_dict(ckp)
        results_dict, test_error, test_auc, test_f1 = summary(model, test_loader, args.n_classes, test_only=True)
        logger('======> Test error: {:.4f}, ROC AUC: {:.4f} Test F1: {:.4f}'.format(test_error, test_auc, test_f1))
        return

    best_test_acc = 0
    best_test_f1 = 0
    start = time.time()
    best_model_save_pth = os.path.join(args.results_dir, "best_%d.pt" % iter)
    last_model_save_pth = os.path.join(args.results_dir, "last_%d.pt"%iter)
    print('Training start')
    for epoch in range(args.max_epochs):
        start = time.time()
        _, _,  _, train_cl_loss = train_loop_clam(args, model, train_loader, optimizer, args.n_classes, args.bag_weight, loss_fn, epoch=epoch)
        print('Train loop use %.3f'%(time.time() - start))
        if args.dfp_discrim:
            mean_instance_per_cls = torch.mean(model.instance_bank, dim=1).detach()
            print('Class 0 confidence bank: ', torch.mean(model.confidence_bank[0]).item())
            print('Class 1 confidence bank: ', torch.mean(model.confidence_bank[1]).item())
            print('Dis between class 0 and 1: ', torch.norm(mean_instance_per_cls[0] - mean_instance_per_cls[1]).item())
        if early_stopping:
            stop = validate_clam(epoch, model, val_loader, args.n_classes, iter,
                                 early_stopping, loss_fn, args.results_dir, logger)

            if stop:
                break

        if epoch % 1 == 0:
            results_dict, test_error, test_auc, test_f1 = summary(model,
                                                                              test_loader,
                                                                              args.n_classes,
                                                                              modelname='clam')
            log_info = '======> @ Epoch {} Test error: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
                                                                                                    test_error,
                                                                                                    test_auc,
                                                                                                    test_f1)
            # if args.dfp_discrim and epoch > 10:
            #     log_info = '======> @ Epoch {} Train CL loss: {:.4f}, Test error: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
            #                                                                                                 train_cl_loss/len(train_loader),
            #                                                                                                 test_error,
            #                                                                                                 test_auc,
            #                                                                                                 test_f1)
            logger(log_info)
            if 1-test_error > best_test_acc or test_f1 > best_test_f1:
                best_test_acc = 1-test_error
                best_test_f1 = test_f1
                torch.save(model.state_dict(), best_model_save_pth)
                torch.save(model.state_dict(), last_model_save_pth)
                logger('******Best Model saved @ %s*******' % last_model_save_pth)
                # if args.dfp_discrim:
                #     dfp_2stage_fts, test_labels, dfp_2stage_fts_stat = get_ft_stat(model, test_split,
                #                                                          type='prompt_2stage')
                #     # fig, axs = plt.subplots(1, 1, figsize=(15, 15))
                #     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                #     features_2d = tsne.fit_transform(dfp_2stage_fts)
                #     plt.scatter(features_2d[:, 0], features_2d[:, 1], c=test_labels,
                #                       cmap=plt.cm.get_cmap("viridis", 2))
                #     plt.tight_layout()
                #     plt.savefig('plots/promptedfeature_tsne_discrim_%s_%s_%s.png'%(args.ft_model, args.mil_method, args.task), dpi=400, bbox_inches='tight')
    if args.early_stopping:
        model.load_state_dict(torch.load(last_model_save_pth))

    results_dict, test_error, test_auc, test_f1 = summary(model, test_loader, args.n_classes)
    logger('======> Test error: {:.4f}, ROC AUC: {:.4f} Test F1: {:.4f}'.format(test_error, test_auc, test_f1))

    # for i in range(args.n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     logger('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    logger('========> [Iter %d] Best: %.3f (Acc), %.3f (F1)'%(iter, best_test_acc, best_test_f1))
    logger('=============================== [Iter %d] Done ===============================' % (iter))

def run_transmil(datasets, model, loss_fn, iter, args, logger):
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if len(test_split) == 0:
        test_split = val_split
        print('[Warning] Testing on valdation set')
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')
    if args.testing:
        ckp_pth = os.path.join(args.results_dir, 'best_%d.pt'%iter)
        print('**** Loading checkpoint from %s ****' % ckp_pth)
        ckp = torch.load(ckp_pth)

        for k, v in ckp.items():
            print(k)
        print('='*10)
        for k, v in model.state_dict().items():
            print(k)

        model.load_state_dict(ckp)
        results_dict, test_error, test_auc, test_f1 = summary(model, test_loader, args.n_classes, modelname='transmil', test_only=True)
        logger('======> Test acc: {:.4f}, ROC AUC: {:.4f} Test F1: {:.4f}'.format(1-test_error, test_auc, test_f1))
        return
    best_test_acc = 0
    best_test_f1 = 0
    start = time.time()
    best_model_save_pth = os.path.join(args.results_dir, "best_%d.pt" % iter)
    last_model_save_pth = os.path.join(args.results_dir, "last_%d.pt" % iter)
    print('Training start')
    for epoch in range(args.max_epochs):
        start = time.time()
        train_loop_transmil(args, model, train_loader, optimizer, args.n_classes, loss_fn)
        print('Train loop use %.3f' % (time.time() - start))
        if epoch % 1 == 0:
            results_dict, test_error, test_auc, test_f1 = summary(model,
                                                                  test_loader,
                                                                  args.n_classes,
                                                                  modelname='transmil')
            logger('======> @ Epoch {} Test error: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
                                                                                                    test_error,
                                                                                                    test_auc,
                                                                                                    test_f1))
            if 1 - test_error > best_test_acc:
                best_test_acc = 1 - test_error
                best_test_f1 = test_f1
                torch.save(model.state_dict(), best_model_save_pth)
                torch.save(model.state_dict(), last_model_save_pth)
                logging.info('Best Model saved @ %s' % last_model_save_pth)

    logger('========> [Iter %d] Best: %.3f (Acc), %.3f (F1)' % (iter, best_test_acc, best_test_f1))
    logger('=============================== [Iter %d] Done ===============================' % (iter))

def run_abmil(datasets, model, loss_fn, iter, args, logger):
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if len(test_split) == 0:
        test_split = val_split
        print('[Warning] Testing on valdation set')
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')
    if args.testing:
        ckp_pth = os.path.join(args.results_dir, 'best_%d.pt'%iter)
        print('**** Loading checkpoint from %s ****' % ckp_pth)
        ckp = torch.load(ckp_pth)

        for k, v in ckp.items():
            print(k)
        print('='*10)
        for k, v in model.state_dict().items():
            print(k)

        model.load_state_dict(ckp)
        results_dict, test_error, test_auc, test_f1 = summary(model, test_loader, args.n_classes, modelname='abmil', test_only=True)
        logger('======> Test error: {:.4f}, ROC AUC: {:.4f} Test F1: {:.4f}'.format(test_error, test_auc, test_f1))
        return
    best_test_acc = 0
    best_test_f1 = 0
    start = time.time()
    best_model_save_pth = os.path.join(args.results_dir, "best_%d.pt" % iter)
    last_model_save_pth = os.path.join(args.results_dir, "last_%d.pt" % iter)
    print('Training start')
    for epoch in range(args.max_epochs):
        start = time.time()
        train_loop_abmil(args, model, train_loader, optimizer, args.n_classes, loss_fn)
        print('Train loop use %.3f' % (time.time() - start))
        if epoch % 1 == 0:
            results_dict, test_error, test_auc, test_f1 = summary(model,
                                                                  test_loader,
                                                                  args.n_classes,
                                                                  modelname='abmil')
            logger('======> @ Epoch {} Test error: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
                                                                                                    test_error,
                                                                                                    test_auc,
                                                                                                    test_f1))
            if 1 - test_error > best_test_acc:
                best_test_acc = 1 - test_error
                best_test_f1 = test_f1
                torch.save(model.state_dict(), best_model_save_pth)
                torch.save(model.state_dict(), last_model_save_pth)
                logging.info('Best Model saved @ %s' % last_model_save_pth)

    logger('========> [Iter %d] Best: %.3f (Acc), %.3f (F1)' % (iter, best_test_acc, best_test_f1))
    logger('=============================== [Iter %d] Done ===============================' % (iter))

def run_dftdmil(datasets, model, loss_fn, iter, args, logger):
    epoch_step = json.loads(args.epoch_step)
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if len(test_split) == 0:
        test_split = val_split
        print('[Warning] Testing on valdation set')
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit optimizer ...', end=' ')
    dimReduction, classifier, attention, attCls = model
    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=args.lr,  weight_decay=args.reg)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=args.lr,  weight_decay=args.reg)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=args.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=args.lr_decay_ratio)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')
    best_test_acc = 0
    best_test_f1 = 0
    start = time.time()
    best_model_save_pth = os.path.join(args.results_dir, "best_%d.pt" % iter)
    last_model_save_pth = os.path.join(args.results_dir, "last_%d.pt" % iter)
    print('Training start')
    for epoch in range(args.max_epochs):
        start = time.time()
        train_loop_dftdmil(dimReduction, classifier, attention, attCls, train_loader,
                         optimizer_adam0, optimizer_adam1, args.n_classes, loss_fn, args,
                         numGroup=args.numGroup, total_instance=args.total_instance, distill=args.distill_type)
        print('Train loop use %.3f' % (time.time() - start))
        if epoch % 1 == 0:
            test_acc, test_auc, test_f1 = test_loop_dftdmil(dimReduction, classifier, attention, attCls, test_loader, loss_fn, args,
                               numGroup=args.numGroup, total_instance=args.total_instance, distill=args.distill_type)
            logger('======> @ Epoch {} Test Acc: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
                                                                                                    test_acc,
                                                                                                    test_auc,
                                                                                                    test_f1))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                dimReduction, classifier, attention, attCls = model
                to_save = {
                    'dimReduction': dimReduction.state_dict(),
                    'classifier': classifier.state_dict(),
                    'attention': attention.state_dict(),
                    'attCls': attCls.state_dict(),
                    }
                torch.save(to_save, best_model_save_pth)
                torch.save(to_save, last_model_save_pth)
                logging.info('Best Model saved @ %s' % last_model_save_pth)
        scheduler0.step()
        scheduler1.step()
    logger('========> [Iter %d] Best: %.3f (Acc), %.3f (F1)' % (iter, best_test_acc, best_test_f1))
    logger('=============================== [Iter %d] Done ===============================' % (iter))

def run_frmil(datasets, model, loss_fn, iter, args, logger):
    epoch_step = json.loads(args.epoch_step)
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if len(test_split) == 0:
        test_split = val_split
        print('[Warning] Testing on valdation set')
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    print('\nInit Loaders...', end=' ')
    train_sampler = CategoriesSampler(train_split.labels,
                                      n_batch=len(train_split.slide_data),
                                      n_cls=args.n_classes,
                                      n_per=1)
    train_loader  = DataLoader(dataset=train_split, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
    val_loader    = DataLoader(dataset=val_split,   batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    print()
    print(model)
    print()

    if args.opt == 'adam':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.reg)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)
    print('Done!')
    if args.testing:
        ckp_pth = os.path.join(args.results_dir, 'best_%d.pt'%iter)
        print('**** Loading checkpoint from %s ****' % ckp_pth)
        ckp = torch.load(ckp_pth)

        for k, v in ckp.items():
            print(k)
        print('='*10)
        for k, v in model.state_dict().items():
            print(k)

        model.load_state_dict(ckp)
        val_loss, val_acc, val_auc, val_f1, val_thrs = test_loop_frmil(model, val_loader, args, set='val')
        # results_dict, val_error, val_auc, val_f1 = summary(model,
        #                                                       val_loader,
        #                                                       args.n_classes,
        #                                                       modelname='frmil',
        #                                                       test_only=True)
        # val_acc = 1 - val_error
        logger('======> @ Test Acc: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(val_acc,
                                                                                      val_auc,
                                                                                      val_f1))
        return
    best_test_acc = 0
    best_test_f1 = 0
    best_model_save_pth = os.path.join(args.results_dir, "best_%d.pt" % iter)
    last_model_save_pth = os.path.join(args.results_dir, "last_%d.pt" % iter)
    print('Training start')
    for epoch in range(args.max_epochs):
        start = time.time()
        train_loop_frmil(epoch, model, train_loader, optimizer, loss_fn, args=args)
        print('Train loop use %.3f' % (time.time() - start))
        if epoch % 1 == 0:
            val_loss, val_acc, val_auc, val_f1, val_thrs = test_loop_frmil(model, val_loader, args, set='val')
            logger('======> 【FRMIL】@ Epoch {} Test Acc: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
                                                                                                    val_acc,
                                                                                                    val_auc,
                                                                                                    val_f1))
            results_dict, val_error, val_auc, val_f1 = summary(model,
                                                               val_loader,
                                                               args.n_classes,
                                                               modelname='frmil',
                                                               test_only=True)
            val_acc = 1 - val_error
            logger('======> 【My】@ Epoch {} Test Acc: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
                                                                                                    val_acc,
                                                                                                    val_auc,
                                                                                                    val_f1))
            if val_acc > best_test_acc:
                best_test_acc = val_acc
                best_test_f1 = val_f1
                torch.save(model.state_dict(), best_model_save_pth)
                torch.save(model.state_dict(), last_model_save_pth)
                logging.info('Best Model saved @ %s' % last_model_save_pth)
        lr_scheduler.step()
    logger('========> [Iter %d] Best: %.3f (Acc), %.3f (F1)' % (iter, best_test_acc, best_test_f1))
    logger('=============================== [Iter %d] Done ===============================' % (iter))

class UnifiedModel(nn.Module):
    def __init__(self, feature_extractor, mil_model, args,
                 bag_weight=0.7, bag_loss=None, ce_weight=None, bce_weight=None):
        super(UnifiedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.args = args
        self.mil_model = mil_model
        self.ce_weight = ce_weight
        self.bce_weight = bce_weight
        if isinstance(self.mil_model, tuple):
            self.mil_model_name = 'DTFDMIL'
            self.dimReduction, self.classifier, self.attention, self.UClassifier = self.mil_model
        else:
            self.mil_model_name = self.mil_model.__class__.__name__
        print(self.mil_model_name)
        self.bag_weight = bag_weight
        self.bag_loss = bag_loss

    def feature_extractor_forward(self, x):
        fts = []
        with torch.no_grad():
            for i in range(len(x)):
                data_i = x[i]
                ft = self.feature_extractor(data_i.unsqueeze(0), return_feature=True)
                fts.append(ft)
        fts = torch.cat(fts, dim=0)
        return fts

    def mil_model_forward(self, ft, label):
        if 'CLAM' in self.mil_model_name:
            logits, Y_prob, Y_hat, _, instance_dict = self.mil_model(ft, label=label, instance_eval=True)
            loss = self.bag_loss(logits, label)
            instance_loss = instance_dict['instance_loss']
            total_loss = self.bag_weight * loss + (1 - self.bag_weight) * instance_loss
        elif self.mil_model_name == 'TransMIL':
            ft = ft.unsqueeze(0)
            results_dict = self.mil_model(data=ft, label=label)
            logits = results_dict['logits']
            Y_hat = results_dict['Y_hat']
            total_loss = self.bag_loss(logits, label)
        elif 'GatedAttention' in self.mil_model_name or 'Attention' in self.mil_model_name:
            # ABMIL
            total_loss, _, Y_hat = self.mil_model.calculate_objective(ft, label)
        elif self.mil_model_name == 'FRMIL':
            if self.mil_model.training:
                norm_idx = torch.where(label == 0)[0].cpu().numpy()[0]
                ano_idx = 1 - norm_idx
                if self.args.drop_data:
                    ft = F.dropout(ft, p=0.20)
                logits, query, max_c = self.mil_model(ft)
                # all losses
                max_c = torch.max(max_c, 1)[0]
                loss_max = F.binary_cross_entropy(max_c, label.float(), weight=self.bce_weight)
                loss_bag = F.cross_entropy(logits, label, weight=self.ce_weight)
                loss_ft = self.bag_loss(query[ano_idx, :, :].unsqueeze(0), query[norm_idx, :, :].unsqueeze(0),
                                   w_scale=query.shape[1])
                total_loss = (loss_bag + loss_ft + loss_max) * (1. / 3)
                Y_hat = torch.argmax(logits, dim=1)
            else:
                logits = self.mil_model(ft)
                total_loss = F.cross_entropy(logits, label.long())
                Y_hat = torch.argmax(logits, dim=1)
        else:
            raise NotImplementedError
        return Y_hat, total_loss

    def forward(self, x, label, train=True, ft_only=False):
        # input [n_patch, image_c, image_h, image_w]

        if len(x.size()) > 4:
            # input [c_cls, n_patch, image_c, image_h, image_w]
            ft = []
            for i in range(x.size(0)):
                ft_i = self.feature_extractor_forward(x[i].squeeze(0))
                ft.append(ft_i.unsqueeze(0))
            ft = torch.cat(ft)
        else:
            ft = self.feature_extractor_forward(x)
        # out_ft [n_patch, feature_dim]
        if ft_only:
            return ft
        if train:
            ft.requires_grad = True
        bag_prediction, total_loss = self.mil_model_forward(ft, label)
        return bag_prediction, total_loss

    def configure_optimizers(self, paras):
        if self.args.opt == "adam":
            optimizer = optim.Adam(paras, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.reg)
        elif self.args.opt == 'adamw':
            optimizer = optim.AdamW(paras, lr=self.args.lr, weight_decay=self.args.reg)
        elif self.args.opt == 'sgd':
            optimizer = optim.SGD(paras, lr=self.args.lr, momentum=0.9, weight_decay=self.args.reg)
        elif self.args.opt == 'radam':
            optimizer = RAdam(paras, lr=self.args.lr, weight_decay=self.args.reg)
        else:
            raise NotImplementedError
        # if self.args.opt == 'adam':
        #     cus_optimizer = torch.optim.Adam(paras, lr=self.args.lr, betas=(0.5, 0.9), weight_decay=self.args.weight_decay)
        # else:
        #     cus_optimizer = torch.optim.AdamW(paras, lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.lr_sch == 'cos':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=5e-6)
        elif self.args.lr_sch == 'step':
            epoch_step = json.loads(self.args.epoch_step)
            sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step,
                                                                gamma=self.args.lr_decay_ratio)
        else:
            raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": sch
        }


EPS = 1e-6
class RGBuvHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=True,
                 hist_boundary=None, green_only=False, device='cuda'):
        """ Computes the RGB-uv histogram feature of a given image.
        Args:
          h: histogram dimension size (scalar). The default value is 64.
          insz: maximum size of the input image; if it is larger than this size, the
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150
            pixels).
          resizing: resizing method if applicable. Options are: 'interpolation' or
            'sampling'. Default is 'interpolation'.
          method: the method used to count the number of pixels for each bin in the
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
          sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
            the sigma parameter of the kernel function. The default value is 0.02.
          intensity_scale: boolean variable to use the intensity scale (I_y in
            Equation 2). Default value is True.
          hist_boundary: a list of histogram boundary values. Default is [-3, 3].
          green_only: boolean variable to use only the log(g/r), log(g/b) channels.
            Default is False.

        Methods:
          forward: accepts input image and returns its histogram feature. Note that
            unless the method is 'thresholding', this is a differentiable function
            and can be easily integrated with the loss function. As mentioned in the
             paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
             training.
        """
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        self.green_only = green_only
        if hist_boundary is None:
            hist_boundary = [-3, 3]
        hist_boundary.sort()
        self.hist_boundary = hist_boundary
        if self.method == 'thresholding':
            self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h
        else:
            self.sigma = sigma

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                          mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                    device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                    device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 1 + int(not self.green_only) * 2,
                             self.h, self.h)).to(device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS),
                                     dim=1)
            else:
                Iy = 1
            if not self.green_only:
                Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] +
                                                                           EPS), dim=1)
                Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] +
                                                                           EPS), dim=1)
                diff_u0 = abs(
                    Iu0 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                diff_v0 = abs(
                    Iv0 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                if self.method == 'thresholding':
                    diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                    diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
                elif self.method == 'RBF':
                    diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u0 = torch.exp(-diff_u0)  # Radial basis function
                    diff_v0 = torch.exp(-diff_v0)
                elif self.method == 'inverse-quadratic':
                    diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                    diff_v0 = 1 / (1 + diff_v0)
                else:
                    raise Exception(
                        f'Wrong kernel method. It should be either thresholding, RBF,'
                        f' inverse-quadratic. But the given value is {self.method}.')
                diff_u0 = diff_u0.type(torch.float32)
                diff_v0 = diff_v0.type(torch.float32)
                a = torch.t(Iy * diff_u0)
                hists[l, 0, :, :] = torch.mm(a, diff_v0)

            Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                                  dim=1)
            Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                                  dim=1)
            diff_u1 = abs(
                Iu1 - torch.unsqueeze(torch.tensor(np.linspace(
                    self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                    dim=0).to(self.device))
            diff_v1 = abs(
                Iv1 - torch.unsqueeze(torch.tensor(np.linspace(
                    self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                    dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == 'inverse-quadratic':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

            diff_u1 = diff_u1.type(torch.float32)
            diff_v1 = diff_v1.type(torch.float32)
            a = torch.t(Iy * diff_u1)
            if not self.green_only:
                hists[l, 1, :, :] = torch.mm(a, diff_v1)
            else:
                hists[l, 0, :, :] = torch.mm(a, diff_v1)

            if not self.green_only:
                Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] +
                                                                           EPS), dim=1)
                Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] +
                                                                           EPS), dim=1)
                diff_u2 = abs(
                    Iu2 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                diff_v2 = abs(
                    Iv2 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                if self.method == 'thresholding':
                    diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
                    diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
                elif self.method == 'RBF':
                    diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u2 = torch.exp(-diff_u2)  # Gaussian
                    diff_v2 = torch.exp(-diff_v2)
                elif self.method == 'inverse-quadratic':
                    diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
                    diff_v2 = 1 / (1 + diff_v2)
                diff_u2 = diff_u2.type(torch.float32)
                diff_v2 = diff_v2.type(torch.float32)
                a = torch.t(Iy * diff_u2)
                hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / (
                ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized