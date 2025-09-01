import os
import time
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support
from sklearn.metrics import auc as calc_auc
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import auc as calc_auc
import torch.nn.functional as F
from model.CLAM.model_clam import CLAM_MB, CLAM_SB
from model.ABMIL.model_abmil import Attention, GatedAttention
from model.TransMIL.model_transmil import TransMIL
from model.FRMIL.model_frmil import FRMIL
from model.prompter import Prompter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.get_feature_extractor import get_extractor
from utils.data_utils import CategoriesSampler
from utils.core_util import define_model, RAdam, get_split_loader, FeatMag
from utils.trainer_util import calculate_error, Meter, compute_accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####Temporarily added#####
def define_clam(args):
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.mil_method != 'mil':
        if 'ResNet50' in args.ft_model:
            model_size = 'small'
        elif 'FPT' in args.ft_model:
            model_size = 'ultra-tiny'
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

def define_model(args):
    if 'CLAM' in args.mil_method:
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

class MetaE2E(nn.Module):
    """Diversity-Aware Prompt Initilization Based on Fast Meta Learning.
    """
    def __init__(self, args, logger=None):
        super(MetaE2E, self).__init__()
        self.args = args
        self.model = define_model(args)
        if 'FPT' in args.ft_model:
            self.frozen_encoder, self.ft_extractor = get_extractor(args)
        else:
            self.ft_extractor = get_extractor(args).to(device)
        self.logger = logger

    def feature_extractor_forward(self, x):
        fts = []
        for i in range(len(x)):
            data_i = x[i]
            ft = self.ft_extractor(data_i.unsqueeze(0), return_feature=True)
            fts.append(ft)
        fts = torch.cat(fts, dim=0)
        return fts

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
                logits = F.softmax(logits, dim=1)[:, 1]

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
        tqdm_gen = tqdm(loader)
        slide_ids = loader.dataset.slide_data['slide_id']
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm_gen):
                image, label = data
                image, label = image.to(device), label.to(device)
                if 'FPT' in self.args.ft_model:
                    with torch.no_grad():
                        _, key_states, value_states = self.frozen_encoder(image, interpolate_pos_encoding=True)
                    ft = self.ft_extractor(image, key_states, value_states)
                else:
                    ft = self.feature_extractor_forward(image)
                # print('ft size:', ft.size())
                if 'CLAM' in modelname:
                        logits, Y_prob, Y_hat, _, results_dict = self.model(ft, label, instance_eval=True,
                                                                       return_features=True)
                elif modelname == 'transmil':
                    prompted_data = ft.unsqueeze(0)
                    results_dict = self.model(data=prompted_data, label=label)
                    Y_prob = results_dict['Y_prob']
                    Y_hat = results_dict['Y_hat']
                elif 'abmil' in modelname:
                    logits, Y_prob, Y_hat, _ = self.model.forward(ft)
                elif 'frmil' in modelname:
                    logits = self.model(ft)
                    Y_prob = F.softmax(logits, dim=1)
                    Y_hat = torch.topk(logits, 1, dim=1)[1]
                else:
                    raise NotImplementedError

                probs = Y_prob.cpu().numpy()
                all_probs[batch_idx] = probs
                all_labels[batch_idx] = label.item()
                all_preds[batch_idx] = Y_hat.item()

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
        return test_error, auc, f1

    def get_optim(self):
        self.prompter_params_gather = []
        self.prompter_params_gather.append(
            {'params': self.ft_extractor.parameters(),
             'lr':self.args.ft_lr,
             'weight_decay':self.args.ft_lr}
        )
        print('Number of trainable parameters in feature extractor: ',
              sum(p.numel() for p in self.ft_extractor.parameters() if p.requires_grad))
        # print the number of parameter that require grad
        for name, param in self.ft_extractor.named_parameters():
            if param.requires_grad:
                print(name)
        self.prompter_params_gather.append(
            {'params': filter(lambda p: p.requires_grad, self.model.parameters()),
             'lr': self.args.lr,
             'weight_decay': self.args.reg}
        )
        print('Number of trainable parameters in MIL model: ',
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))

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

    def clam_runner(self, feature, label, loss_fn):
        logits, Y_prob, Y_hat, _, instance_dict = self.model(feature, label=label, instance_eval=True)
        loss = loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = self.args.bag_weight * loss + (1 - self.args.bag_weight) * instance_loss
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def transmil_runner(self, feature, label, loss_fn):
        feature = feature.unsqueeze(0)
        results_dict = self.model(data=feature, label=label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        total_loss = loss_fn(logits, label)
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def abmil_runner(self, feature, label, loss_fn):
        logits, Y_prob, Y_hat, A = self.model.forward(feature)
        total_loss = loss_fn(logits, label)
        error = 1. - Y_hat.eq(label).cpu().float().mean().item()
        return total_loss, error

    def frmil_runner(self, feature, label, loss_fn, bce_weight, ce_weight):
        norm_idx = torch.where(label.cpu() == 0)[0].numpy()[0]
        ano_idx = 1 - norm_idx
        if self.args.drop_data:
            feature = F.dropout(feature, p=0.20)
        logits, query, max_c = self.model(feature)

        # all losses
        max_c = torch.max(max_c, 1)[0]
        loss_max = F.binary_cross_entropy(max_c, label.float(), weight=bce_weight)
        loss_bag = F.cross_entropy(logits, label, weight=ce_weight)
        loss_ft = loss_fn(query[ano_idx, :, :].unsqueeze(0), query[norm_idx, :, :].unsqueeze(0),
                           w_scale=query.shape[1])
        loss = (loss_bag + loss_ft + loss_max) * (1. / 3)
        acc = compute_accuracy(logits, label)

        return loss, 1 - acc/100

    def forward(self, iter, datasets):
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
            test_loader = DataLoader(dataset=val_split, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
        else:
            train_loader = get_split_loader(train_split, training=True, weighted=self.args.weighted_sample)
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

            ft_ckp_pth = os.path.join(self.args.results_dir, 'best_feature_extractor_%d.pt' % iter)
            print('**** Loading prompt checkpoint from %s ****' % ft_ckp_pth)
            ft_ckp = torch.load(ft_ckp_pth)
            self.ft_extractor.load_state_dict(ft_ckp)
            # if 'frmil' in self.args.mil_method:
            #     _, test_acc, test_auc, test_f1, test_thrs = self.summary_frmil(test_loader)
            # else:
            test_error, test_auc, test_f1 = self.summary(test_loader,
                                                       self.args.n_classes,
                                                       modelname=self.args.mil_method)
            test_acc = 1 - test_error
            self.logger('======> Test acc: {:.4f}, ROC AUC: {:.4f} Test F1: {:.4f}'.format(test_acc, test_auc, test_f1))
            return

        best_test_acc = 0
        best_test_f1 = 0
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        best_feature_extractor_save_pth = os.path.join(self.args.results_dir, "best_feature_extractor_%d.pt" % iter)
        print('Training start')
        for epoch in range(self.args.max_epochs):
            start = time.time()
            self.model.train()
            tqdm_gen = tqdm(train_loader)
            train_loss = 0.
            train_error = 0.

            if 'frmil' in self.args.mil_method:
                ce_weight = [i for i in train_loader.dataset.count_dict.values()]
                ce_weight = 1. / torch.tensor(ce_weight, dtype=torch.float)
                ce_weight = ce_weight.cuda()
                bce_weight = train_loader.dataset.pos_weight.cuda()

            for batch_idx, data in enumerate(tqdm_gen):
                global_step = len(train_loader) * epoch + batch_idx
                scheduler(global_step)
                image, label = data
                image, label = image.to(device), label.to(device)
                if 'FPT' in self.args.ft_model:
                    with torch.no_grad():
                        _, key_states, value_states = self.frozen_encoder(image, interpolate_pos_encoding=True)
                    ft = self.ft_extractor(image, key_states, value_states)
                else:
                    ft = self.feature_extractor_forward(image)
                # print('ft size:', ft.size())
                if 'CLAM' in self.args.mil_method:
                    total_loss, error = self.clam_runner(ft, label, loss_fn)
                elif 'transmil' in self.args.mil_method:
                    total_loss, error = self.transmil_runner(ft, label, loss_fn)
                elif 'abmil' in self.args.mil_method:
                    total_loss, error = self.abmil_runner(ft, label, loss_fn)
                elif 'frmil' in self.args.mil_method:
                    total_loss, error = self.frmil_runner(ft, label, loss_fn, bce_weight, ce_weight)
                else:
                    raise NotImplementedError
                train_error += error
                total_loss.backward()
                loss_value = total_loss.item()
                train_loss += loss_value
                # step
                if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # calculate loss and error for epoch
            train_loss /= len(train_loader)
            train_error /= len(train_loader)

            if epoch % 1 == 0:
                # if 'frmil' in self.args.mil_method:
                #     _, test_acc, test_auc, test_f1, test_thrs = self.summary_frmil(test_loader)
                # else:
                test_error, test_auc, test_f1 = self.summary(test_loader,
                                                            self.args.n_classes,
                                                            modelname=self.args.mil_method)
                test_acc = 1 - test_error
                log_info = '======> @ Epoch {} Test acc: {:.4f}, ROC AUC: {:.4f}, Test F1: {:.4f}'.format(epoch,
                                                                                                            test_acc,
                                                                                                            test_auc,
                                                                                                            test_f1)

                self.logger(log_info)
                if test_acc > best_test_acc and test_f1 > best_test_f1:
                    best_test_acc = test_acc
                    best_test_f1 = test_f1
                    torch.save(self.model.state_dict(), best_model_save_pth)
                    torch.save(self.ft_extractor.state_dict(), best_feature_extractor_save_pth)
                    self.logger('******Best Model saved @ %s*******' % best_model_save_pth)

        self.logger('========> [Iter %d] Best: %.3f (Acc), %.3f (F1)' % (iter, best_test_acc, best_test_f1))
        self.logger('=============================== [Iter %d] Done ===============================' % (iter))