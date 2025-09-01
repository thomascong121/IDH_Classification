import sys

from torch import nn

sys.path.insert(1, '/scratch/sz65/cc0395/WSI_prompt/')
import os
import numpy as np
import torch
import random
import math
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.metrics import auc as calc_au
import torch.nn.functional as F

def five_scores(bag_labels, bag_predictions,sub_typing=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    avg = 'macro' if sub_typing else 'binary'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    return accuracy, auc_value, precision, recall, fscore
def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps
def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error
def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.

def compute_accuracy_bce(logits, labels, thr=0.5):
    pred = torch.ge(logits, thr).float()
    return (pred == labels).type(torch.float).mean().item() * 100.
def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal
def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc
def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx  = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions, op_thres=None):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    if op_thres is not None:
        this_class_label[this_class_label>=op_thres] = 1
        this_class_label[this_class_label<op_thres] = 0
    else:
        this_class_label[this_class_label>=threshold_optimal] = 1
        this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='macro')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore, threshold_optimal

class Meter(object):

    def __init__(self):
        self.list = []
        self.labels = []
        self.preds = []

    def update(self, item):
        self.list.append(item)

    def update_gt(self, label, pred):
        self.labels.append(np.clip(label, 0, 1))
        self.preds.append(pred)

    def avg_test(self):
        return torch.tensor(np.array(five_scores(self.labels, self.preds)[0])) * 100.0

    def acc_auc(self, thres=None):
        accuracy, auc_value, precision, recall, fscore, thres_op = five_scores(self.labels, self.preds, op_thres=thres)
        acc = torch.tensor(np.array(accuracy)) * 100.0
        auc = torch.tensor(np.array(auc_value)) * 100.0
        return acc, auc, fscore, thres_op

    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

    def std(self):
        return torch.tensor(self.list).std() if len(self.list) else None

    def confidence_interval(self):
        if len(self.list) == 0:
            return None
        std = torch.tensor(self.list).std()
        ci = std * 1.96 / math.sqrt(len(self.list))
        return ci

    def avg_and_confidence_interval(self):
        return self.avg(), self.confidence_interval()

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def forward(self, instance, instance_bank, instance_label):
        # given a batch of instance, minimize the distance between the instance and the mean instance in the bank which has the same label
        # and maximize the distance between the instance and the mean instance in the bank which has different label
        # instance: [N, D]
        # instance_bank: [C, M, D]
        # instance_label: [N]
        # C: number of classes
        # M: number of instances per class in the memory bank
        # D: feature dimension
        # N: batch size
        # print(instance.size(), instance_bank.size(), instance_label.size())
        mean_instance_per_cls = torch.mean(instance_bank, dim=1).detach()  # [C, D]
        pos_mask = instance_label.unsqueeze(1) == torch.arange(mean_instance_per_cls.size(0)).to(mean_instance_per_cls.device).unsqueeze(0)  # [N, C]
        neg_mask = ~pos_mask
        pos_mean_instance = torch.matmul(pos_mask.float(), mean_instance_per_cls)  # [N, D]
        neg_mean_instance = torch.matmul(neg_mask.float(), mean_instance_per_cls)  # [N, D]
        pos_dist = torch.nn.functional.pairwise_distance(instance, pos_mean_instance)
        neg_dist = torch.nn.functional.pairwise_distance(instance, neg_mean_instance)
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(instance, pos_mean_instance.T),
        #     0.7)
        # anchor_dot_contrast_neg = torch.div(
        #     torch.matmul(instance, neg_mean_instance.T),
        #     0.7)
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        # log_prob = logits - torch.log(torch.exp(anchor_dot_contrast+ anchor_dot_contrast_neg) + 1e-12)
        # loss = - log_prob

        # loss = torch.pow(pos_dist, 2) + torch.pow(torch.clamp(self.m - neg_dist, min=0.0), 2)
        # loss = loss.mean()

        loss = self.triplet(instance, pos_mean_instance, neg_mean_instance)
        return {'loss': loss, 'pos_dist': pos_dist.mean(), 'neg_dist': neg_dist.mean()}

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim    = feat_dim
        self.device      = device

        center_init = torch.zeros(self.num_classes, self.feat_dim).to(self.device)

        nn.init.xavier_uniform_(center_init)
        self.centers = nn.Parameter(center_init)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long() # should be long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask   = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    def get_assignment(self, batch):
        alpha = 1.0
        norm_squared = torch.sum((batch.unsqueeze(1) - self.centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / alpha))
        power = float(alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self, batch):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

def train_loop_clam(args, model, loader, optimizer, n_classes, bag_weight, loss_fn = None, n_data=0, epoch=0):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_cl_loss = 0.
    train_inst_loss = 0.
    inst_count = 0
    num_exp = 0
    pos_dist = 0.
    neg_dist = 0.
    instance_cl_loss = ContrastiveLoss(m=2)
    center_loss = CenterLoss(num_classes=2,
                              feat_dim=model.embed_length,
                              device=device)

    # if args.dfp_discrim and epoch == args.prompt_epoch:
    #     print('************Start using prompt************')
    #     old_params_names = [name for name, param in model.named_parameters() if param.requires_grad]
    #     model.prompt_init()
    #     prompt_params_names = [name for name, param in model.named_parameters() if param.requires_grad]
    #     new_params = [params for name, params in model.named_parameters() if params.requires_grad and name not in old_params_names]
    #     print('Old params: ', len(old_params_names), 'Prompt params: ', len(prompt_params_names))
    #     print('New params: ', new_params)
    #     optimizer.add_param_group({'params': new_params})
    #     print('************Prompt initialized************')
    #     use_prompt = True
    # elif args.dfp_discrim and epoch > args.prompt_epoch:
    #     use_prompt = True
    #     print('New params: ', model.prompt_embeddings.data)
    # else:
    #     use_prompt = False
    use_prompt_test = False

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True, use_prompt=use_prompt_test)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss
        # print('Loss calculating ', time.time() - time2)
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        error = calculate_error(Y_hat, label)
        train_error += error
        total_loss.backward()
        loss_value = total_loss.item()
        train_loss += loss_value
        # step
        if (batch_idx + 1) % args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

        num_exp += len(label)
        if (n_data > 0) and (num_exp >= n_data):
            break

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    # pos_dist /= len(loader)
    # neg_dist /= len(loader)
    # print('pos dist: ', pos_dist, 'neg dist: ', neg_dist)
    return train_loss, train_inst_loss,  train_error, train_cl_loss

def validate_clam(epoch, model, loader, n_classes, iter, early_stopping = None, loss_fn = None, results_dir = None, logger=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']

            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
    logger('Val Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc), print_=False)
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "last_%d.pt"%iter))

        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False

def train_loop_transmil(args, model, loader, optimizer, n_classes, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        data = data.unsqueeze(0)
        results_dict = model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        loss.backward()
        # step
        if (batch_idx + 1) % args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    return train_loss,  train_error

def train_loop_abmil(args, model, loader, optimizer, n_classes, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # calculate loss and metrics
        logits, Y_prob, Y_hat, A = model.forward(data)
        loss = loss_fn(logits, label)
        train_loss += loss.item()
        # error, _ = model.calculate_classification_error(data, label)
        error = 1. - Y_hat.eq(label).cpu().float().mean().item()
        train_error += error
        # backward pass
        loss.backward()
        # step
        if (batch_idx + 1) % args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    return train_loss,  train_error

def dftd_train_forward(dimReduction, classifier, attention, UClassifier,
                  optimizer0, optimizer1, loss_fn, args, batch_idx,
                  data, label, numGroup, total_instance, device, distill):
    slide_pseudo_feat = []
    slide_sub_preds = []
    slide_sub_labels = []
    tslideLabel = label.unsqueeze(0)
    # random split data into numGroup
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    instance_per_group = total_instance // numGroup

    for tindex in index_chunk_list:
        slide_sub_labels.append(tslideLabel)
        subFeat_tensor = torch.index_select(data, dim=0, index=torch.LongTensor(tindex).to(device))
        tmidFeat = dimReduction(subFeat_tensor)
        tAA = attention(tmidFeat).squeeze(0)
        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
        slide_sub_preds.append(tPredict)

        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
        topk_idx_max = sort_idx[:instance_per_group].long()
        topk_idx_min = sort_idx[-instance_per_group:].long()
        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

        MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
        max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
        af_inst_feat = tattFeat_tensor

        if distill == 'MaxMinS':
            slide_pseudo_feat.append(MaxMin_inst_feat)
        elif distill == 'MaxS':
            slide_pseudo_feat.append(max_inst_feat)
        elif distill == 'AFS':
            slide_pseudo_feat.append(af_inst_feat)
    slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
    ## optimization for the first tier
    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup
    loss0 = loss_fn(slide_sub_preds, slide_sub_labels.squeeze(-1)).mean()

    loss0.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), args.grad_clipping)
    torch.nn.utils.clip_grad_norm_(attention.parameters(), args.grad_clipping)
    torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clipping)

    ## optimization for the second tier
    gSlidePred = UClassifier(slide_pseudo_feat)
    loss1 = loss_fn(gSlidePred, tslideLabel.squeeze(-1)).mean()
    loss1.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), args.grad_clipping)

    if (batch_idx + 1) % args.accumulate_grad_batches == 0:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        optimizer0.step()
        optimizer1.step()

    train_loss_t1 = loss0.item() * len(index_chunk_list)
    train_loss_t2 = loss1.item()
    pred = torch.argmax(torch.softmax(gSlidePred, dim=1))
    return train_loss_t1,  train_loss_t2, pred, len(index_chunk_list)

def dftd_test_forward(dimReduction, classifier, attention, UClassifier,
                  gPred_0, gt_0, gPred_1, gt_1, args,
                  data, label, numGroup, total_instance, device, distill):
    tslideLabel = label.unsqueeze(0)
    midFeat = dimReduction(data)
    AA = attention(midFeat, isNorm=False).squeeze(0)  ## N
    allSlide_pred_softmax = []
    instance_per_group = total_instance // numGroup
    for jj in range(args.num_MeanInference):
        feat_index = list(range(data.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        slide_d_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        for tindex in index_chunk_list:
            slide_sub_labels.append(tslideLabel)
            idx_tensor = torch.LongTensor(tindex).to(device)
            tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

            tAA = AA.index_select(dim=0, index=idx_tensor)
            tAA = torch.softmax(tAA, dim=0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

            tPredict = classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

            if distill == 'MaxMinS':
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif distill == 'MaxS':
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx = topk_idx_max
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif distill == 'AFS':
                slide_d_feat.append(tattFeat_tensor)

        slide_d_feat = torch.cat(slide_d_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

        gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
        gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)

        gSlidePred = UClassifier(slide_d_feat)
        allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

    allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
    allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
    gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
    gt_1 = torch.cat([gt_1, tslideLabel], dim=0)
    return gPred_0, gt_0, gPred_1, gt_1
def train_loop_dftdmil(dimReduction, classifier, attention, UClassifier, loader,
                         optimizer0, optimizer1, n_classes, loss_fn, args,
                         numGroup=3, total_instance=3, distill='MaxMinS'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()
    train_loss_t1 = 0.
    train_loss_t2 = 0.
    number_ins_per_group = []
    for batch_idx, (data, label) in enumerate(loader):
        # data: nxemb_lgh, label: n
        data, label = data.to(device), label.to(device)
        loss_t1, loss_t2, _, count = dftd_train_forward(dimReduction, classifier, attention, UClassifier,
                                          optimizer0, optimizer1, loss_fn, args, batch_idx,
                                          data, label, numGroup, total_instance, device, distill)
        number_ins_per_group.append(data.size(0))
        train_loss_t1 += loss_t1
        train_loss_t2 += loss_t2
    print('Max/min number of instances per group: ', max(number_ins_per_group), min(number_ins_per_group))
    # calculate loss and error for epoch
    train_loss_t1 /= (len(loader) * count)
    train_loss_t2 /= len(loader)
    return train_loss_t1,  train_loss_t2

def test_loop_dftdmil(dimReduction, classifier, attention, UClassifier, loader, loss_fn, args,
                         numGroup=3, total_instance=3, distill='MaxMinS'):
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gPred_0 = torch.FloatTensor().to(device)
    gt_0 = torch.LongTensor().to(device)
    gPred_1 = torch.FloatTensor().to(device)
    gt_1 = torch.LongTensor().to(device)

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            # data: nxemb_lgh, label: n
            data, label = data.to(device), label.to(device)
            gPred_0, gt_0, gPred_1, gt_1 = dftd_test_forward(dimReduction, classifier, attention, UClassifier,
                                          gPred_0, gt_0, gPred_1, gt_1, args,
                                          data, label, numGroup, total_instance, device, distill)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)

    print(f'  First-Tier acc {macc_0}, F1 {mF1_0}, AUC {auc_0}')
    print(f'  Second-Tier acc {macc_1}, F1 {mF1_1}, AUC {auc_1}')
    return macc_1, auc_1, mF1_1

def train_loop_frmil(epoch, model, loader, optimizer, loss_fn, args=None):
    model.train()

    loss_meter = Meter()
    acc_meter = Meter()
    tqdm_gen = tqdm(loader)
    ce_weight = [i for i in loader.dataset.count_dict.values()]
    ce_weight = 1. / torch.tensor(ce_weight, dtype=torch.float)
    ce_weight = ce_weight.cuda()
    bce_weight = loader.dataset.pos_weight.cuda()

    # $\tau$ predefined using feature analysis
    # CM16 (simclr) --> 8.48
    # MSI  (imgnet) --> 52.5

    mag_loss = loss_fn

    for batch_idx, (data, labels) in enumerate(tqdm_gen):
        # Index of Normal Bags in Batch [N,K,C].
        norm_idx = torch.where(labels == 0)[0].numpy()[0]
        ano_idx = 1 - norm_idx

        data, labels = data.cuda(), labels.cuda().long()


        if args.drop_data:
            data = F.dropout(data, p=0.20)
        # print('ft size ', data.size())
        logits, query, max_c = model(data)

        # all losses
        max_c = torch.max(max_c, 1)[0]
        loss_max = F.binary_cross_entropy(max_c, labels.float(), weight=bce_weight)
        loss_bag = F.cross_entropy(logits, labels, weight=ce_weight)
        loss_ft = mag_loss(query[ano_idx, :, :].unsqueeze(0), query[norm_idx, :, :].unsqueeze(0), w_scale=query.shape[1])
        loss = (loss_bag + loss_ft + loss_max) * (1. / 3)

        acc = compute_accuracy(logits, labels)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        if (batch_idx + 1) % args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()
        # optimizer.step()
        # optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.std()

def test_loop_frmil(model, loader, args=None, set='val', thrs=0.5):
    model.eval()

    loss_meter = Meter()
    acc_meter  = Meter()

    tqdm_gen = tqdm(loader)

    with torch.no_grad():
        for _, (data, labels) in enumerate(tqdm_gen, 1):

            data   = data.cuda()
            labels = labels.cuda().float()

            logits = model(data)
            loss   = F.cross_entropy(logits, labels.long())

            acc    = compute_accuracy(logits, labels.long())
            logits = F.softmax(logits,dim=1)[:,1]

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            acc_meter.update_gt(labels.data.cpu().numpy()[0],logits.data.squeeze().cpu().numpy())


    if set == 'val':
        acc, auc, fscore, op_thrs = acc_meter.acc_auc()
        return loss_meter.avg(), acc, auc, fscore, op_thrs
    else:
        acc, auc, fscore, op_thrs = acc_meter.acc_auc(thrs)
        return loss_meter.avg(), acc, auc, fscore