import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import sklearn.cluster as cluster
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def gen_csv(slide_dict, out_root, name):
    df_slide = pd.DataFrame.from_dict(slide_dict)
    df_slide.to_csv(out_root + '/%s'%name, index=False)

def gen_color_proto():
    SKIP = ['.DS_Store', '._.DS_Store']
    data_root = os.getenv('PBS_JOBFS')
    img_root = data_root + '/BRIGHT/'
    local_root = 'data/pre_extracted_color_feature/BRIGHT/Train/'

    hc = cluster.AgglomerativeClustering(
        n_clusters=None,
        linkage='average',
        distance_threshold=0.8,
    )

    if not os.path.exists(local_root):
        os.makedirs(local_root)

    label_list = os.listdir(img_root + '/train')
    rep_gather = None
    feature_list = []
    for top_label in label_list:
        if top_label in SKIP:
            continue
        slide_label_folder = os.listdir(img_root + '/train/' + top_label)
        for slide_label in slide_label_folder:
            if slide_label in SKIP:
                continue
            slide_pth = os.listdir(img_root + '/train/' + top_label + '/' + slide_label + '/Color_features/pt_files/')
            for pt in tqdm(slide_pth):
                if pt == SKIP:
                    continue
                base_name = pt.split('.')[0]
                color_pt_path = img_root + '/train/' + top_label + '/' + slide_label + '/Color_features/pt_files/' + base_name + '.pt'
                color_ft = torch.load(color_pt_path)
                rand_idx = torch.randperm(color_ft.size(0))[:50]
                feature_for_rep = color_ft[rand_idx].detach().cpu()
                # feature_for_rep = color_ft.detach().cpu()
                rep_gather = feature_for_rep if rep_gather is None else torch.cat([rep_gather, feature_for_rep], dim=0)
    print('Feature shape:', rep_gather.size())
    y_pred = hc.fit(rep_gather.numpy()).labels_
    y_pred = torch.from_numpy(y_pred).to(device)
    coarse_class_idx = torch.unique(y_pred)
    num_coarse_classes = len(coarse_class_idx)
    print("Nums of coarsely divided categories for dataset: {}".format(num_coarse_classes))
    prototype_gather = []
    for i in range(len(coarse_class_idx)):
        pos = torch.where(y_pred == i)[0]
        prototype = rep_gather[pos].mean(0).unsqueeze(0)
        prototype_gather.append(prototype)
    prototype_gather = torch.cat(prototype_gather)
    # # # #save tensor
    print('Prototype size:', prototype_gather.size())
    torch.save(prototype_gather, os.path.join(local_root, 'prototype.pt'))

def get_stain_idx(stain_proto, rep_batch):
    rep_batch_sum = (rep_batch ** 2).sum(dim=-1, keepdims=True)  # [N, 1]
    prototype_gather_sum = (stain_proto ** 2).sum(dim=-1, keepdims=True).T  # [1, M]            #### TODO ####
    distance_matrix = torch.sqrt(
        rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, stain_proto.T))  # [N, M]
    indices = torch.argmin(distance_matrix, dim=-1)
    return indices
def gather_csv(project_id, feature_dir):
    local_root_stain_proto = 'data/pre_extracted_color_feature/BRIGHT/Train/'
    SKIP = ['.DS_Store', '._.DS_Store', '.', '..']
    if 'HIPT' in feature_dir:
        data_root = 'BRIGHT_HIPT_patches'
    else:
        data_root = 'BRIGHT_patches'
    img_root = f'/g/data/{project_id}/{data_root}/'
    out_root = f'/g/data/{project_id}/{data_root}/{feature_dir}'
    color_root = f'/g/data/{project_id}/BRIGHT_patches/'
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    # local_root = f'/scratch/{project_id}/cc0395/WSI_Image/BRIGHT'
    train_slide = {'slide_id': [], 'label': []}
    test_slide = {'slide_id': [], 'label': []}
    stain_proto = torch.load('%s/prototype.pt'%local_root_stain_proto)

    print('Loaded stain prototype:', stain_proto.size())
    for stage in ['train', 'test']:
        if stage == SKIP:
            continue
        label_list = os.listdir(img_root + stage)

        for top_label in label_list:
            if top_label in SKIP:
                continue
            slide_label_folder = os.listdir(img_root + stage + '/' + top_label)
            for slide_label in slide_label_folder:
                if slide_label in SKIP:
                    continue
                slides = os.listdir(img_root + stage + '/' + top_label + '/' + slide_label + f'/{feature_dir}/pt_files/')
                stain_label = img_root + stage + '/' + top_label + '/' + slide_label + '/Color_label/pt_files/'
                if not os.path.exists(stain_label):
                    os.makedirs(stain_label)
                for slide in slides:
                    if slide == SKIP or slide.startswith('._'):
                        continue
                    stain_path = color_root + stage + '/' + top_label + '/' + slide_label + '/Color_features/pt_files/' + slide
                    slide_stain_path = stain_label + slide
                    stain_ft = torch.load(stain_path)
                    if len(stain_ft.size()) > 2:
                        b, n, _ = stain_ft.size()
                        stain_ft = stain_ft.view(-1, stain_ft.size(-1))
                    indices = get_stain_idx(stain_proto, stain_ft)
                    torch.save(indices, slide_stain_path)
                    # if '.pt' in slide:
                    #     slide = slide.split('.')[0]
                    slide_path = img_root + stage + '/' + top_label + '/' + slide_label + f'/{feature_dir}/pt_files/' + slide
                    if stage == 'train':
                        train_slide['slide_id'].append(slide_path)
                        train_slide['label'].append(top_label)
                    else:
                        test_slide['slide_id'].append(slide_path)
                        test_slide['label'].append(top_label)
    print('====Train===')
    print('Number of train slides:', len(train_slide['slide_id']))
    print('Number of Cancerous slides:', train_slide['label'].count('Cancerous'))
    print('Number of Non-cancerous slides:', train_slide['label'].count('Non-cancerous'))
    print('Number of Pre-cancerous slides:', train_slide['label'].count('Pre-cancerous'))
    print('====Test===')
    print('Number of test slides:', len(test_slide['slide_id']))
    print('Number of Cancerous slides:', test_slide['label'].count('Cancerous'))
    print('Number of Non-cancerous slides:', test_slide['label'].count('Non-cancerous'))
    print('Number of Pre-cancerous slides:', test_slide['label'].count('Pre-cancerous'))
    gen_csv(train_slide, out_root, 'train_slide.csv')
    gen_csv(test_slide, out_root, 'test_slide.csv')
    print('CSV files generated @', out_root)


parser = argparse.ArgumentParser(description='Dataset Preprocess')
parser.add_argument('--feat_dir', type=str, default='R50_features')
args = parser.parse_args()
if __name__ == '__main__':
    # gen_color_proto()
    project_id = 'iq24'
    gather_csv(project_id, args.feat_dir)
