import argparse
import os
import cv2
import h5py
import pandas as pd
import shutil
import torch
from tqdm import tqdm
import sklearn.cluster as cluster
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def gen_csv(slide_dict, out_root, name):
    df_slide = pd.DataFrame.from_dict(slide_dict)
    df_slide.to_csv(out_root + '/%s'%name, index=False)

def gen_color_proto(project_id):
    SKIP = ['.DS_Store', '._.DS_Store']
    data_root = f'/g/data/{project_id}'
    img_root = data_root + '/CAMELYON16_patches/'
    local_root = 'data/pre_extracted_color_feature/CAM16/Train/'

    hc = cluster.AgglomerativeClustering(
        n_clusters=None,
        linkage='average',
        distance_threshold=0.9,
    )

    if not os.path.exists(local_root):
        os.makedirs(local_root)

    label_list = os.listdir(img_root + '/train')
    rep_gather = None
    for slide_label in label_list:
        if slide_label in SKIP:
            continue
        slides_pt = os.listdir(img_root + '/train' + '/' + slide_label + '/Color_features/pt_files/')
        for pt in tqdm(slides_pt):
            if pt == SKIP:
                continue
            base_name = pt.split('.')[0]
            color_pt_path = img_root + '/train' + '/' + slide_label + '/Color_features/pt_files/' + base_name + '.pt'
            color_ft = torch.load(color_pt_path)
            rand_idx = torch.randperm(color_ft.size(0))[:50]
            feature_for_rep = color_ft[rand_idx].detach().cpu()
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
    torch.save(prototype_gather, os.path.join(local_root, 'prototype_all.pt'))

def get_stain_idx(stain_proto, rep_batch):
    rep_batch_sum = (rep_batch ** 2).sum(dim=-1, keepdims=True)  # [N, 1]
    prototype_gather_sum = (stain_proto ** 2).sum(dim=-1, keepdims=True).T  # [1, M]            #### TODO ####
    distance_matrix = torch.sqrt(
        rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, stain_proto.T))  # [N, M]
    indices = torch.argmin(distance_matrix, dim=-1)
    return indices
def gather_csv(project_id, feature_dir):
    local_root_stain_proto = 'data/pre_extracted_color_feature/CAM16/Train/'
    SKIP = ['.DS_Store', '._.DS_Store']
    # csv_root = os.getenv('PBS_JOBFS')
    # local_root = '/scratch/sz65/cc0395/WSI_Image/unitopath-public'
    img_root = f'/g/data/{project_id}/CAMELYON16_patches/'#csv_root + '/CAM16/'
    out_root = f'/g/data/{project_id}/CAMELYON16_patches/{feature_dir}'
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    # rt = os.environ.get('PBS_JOBFS')
    # local_root = f'/{rt}/CAM16'
    train_slide = {'slide_id': [], 'label': []}
    test_slide = {'slide_id': [], 'label': []}
    stain_proto = torch.load('%s/prototype.pt'%local_root_stain_proto)

    print('Loaded stain prototype:', stain_proto.size())
    for stage in ['train', 'test']:
        if stage == SKIP:
            continue
        label_list = os.listdir(img_root + stage)
        for slide_label in label_list:
            if slide_label in SKIP:
                continue
            slides = os.listdir(img_root + stage + '/' + slide_label + f'/{feature_dir}/pt_files/')
            stain_label = img_root + stage + '/' + slide_label + '/Color_label/pt_files/'
            if not os.path.exists(stain_label):
                os.makedirs(stain_label)
            for slide in slides:
                if slide == SKIP:
                    continue
                slide_path = img_root + stage + '/' + slide_label + f'/{feature_dir}/pt_files/' + slide
                stain_path = img_root + stage + '/' + slide_label + '/Color_features/pt_files/' + slide
                slide_stain_path = stain_label + slide
                stain_ft = torch.load(stain_path)
                if len(stain_ft.size()) > 2:
                    b, n, _ = stain_ft.size()
                    stain_ft = stain_ft.view(-1, stain_ft.size(-1))
                indices = get_stain_idx(stain_proto, stain_ft)
                # print(indices)
                torch.save(indices, slide_stain_path)
                if stage == 'train':
                    train_slide['slide_id'].append(slide_path)
                    train_slide['label'].append(slide_label)
                else:
                    test_slide['slide_id'].append(slide_path)
                    test_slide['label'].append(slide_label)
    print('====Train===')
    print('Number of train slides:', len(train_slide['slide_id']))
    print('Number of normal slides:', train_slide['label'].count('normal'))
    print('Number of tumor slides:', train_slide['label'].count('tumor'))
    print('====Test===')
    print('Number of test slides:', len(test_slide['slide_id']))
    print('Number of normal slides:', test_slide['label'].count('normal'))
    print('Number of tumor slides:', test_slide['label'].count('tumor'))
    gen_csv(train_slide, out_root, 'train_slide.csv')
    gen_csv(test_slide, out_root, 'test_slide.csv')
    print('CSV saved to:', out_root)

# def colour_feature_transform(ft_np_pth, ft_tensor_pth):
def pt_file_regenerate(feature_type='R50_features'):
    SKIP = ['.DS_Store', '._.DS_Store']
    csv_root = os.getenv('PBS_JOBFS')
    img_root = csv_root + '/CAM16/'

    for stage in ['train', 'test']:
        if stage == SKIP:
            continue
        label_list = os.listdir(img_root + stage)
        for slide_label in tqdm(label_list):
            if slide_label in SKIP:
                continue
            r50_color_pth = img_root + stage + '/' + slide_label + '/R50_Color_features/pt_files/'
            if not os.path.exists(r50_color_pth):
                os.makedirs(r50_color_pth)
            slides_pt = os.listdir(img_root + stage + '/' + slide_label + '/R50_features/h5_files/')
            for pt in slides_pt:
                if pt == SKIP:
                    continue
                base_name = pt.split('.')[0]
                r50_pt_path = img_root + stage + '/' + slide_label + '/R50_features/h5_files/' + pt
                color_pt_path = img_root + stage + '/' + slide_label + '/Color_features/h5_files/' + base_name + '.h5'
                with h5py.File(r50_pt_path, 'r') as hdf5_file:
                    r50_ft = hdf5_file['features'][:]
                    r50_ft = torch.from_numpy(r50_ft)
                with h5py.File(color_pt_path, 'r') as hdf5_file:
                    color_ft = hdf5_file['features'][:]
                    color_ft = torch.from_numpy(color_ft)
                # r50_ft = torch.load(r50_pt_path)
                # color_ft = torch.load(color_pt_path)
                # print('size check ', r50_ft.size(), color_ft.size())
                if r50_ft.size(0) != color_ft.size(0):
                    print('Size mismatch:', r50_ft.size(), color_ft.size())
                    min_size = min(r50_ft.size(0), color_ft.size(0))
                    r50_color_ft = torch.cat((r50_ft[:min_size, :], color_ft[:min_size, :]), dim=1)
                else:
                    r50_color_ft = torch.cat((r50_ft, color_ft), dim=1)
                torch.save(r50_color_ft, r50_color_pth + base_name + '.pt')
                # slides_h5 = os.listdir(img_root + stage + '/' + slide_label + '/%s/h5_files/'%feature_type)
            # for h5_f in slides_h5:
            #     if h5_f == SKIP:
            #         continue
            #     base_name = h5_f.split('.')[0]
            #     slide_pt_path = img_root + stage + '/' + slide_label + '/%s/h5_files/'%feature_type + h5_f
            #     slide_pt_path_new = img_root + stage + '/' + slide_label + '/%s/pt_files/'%feature_type + base_name + '.pt'
            #     with h5py.File(slide_pt_path, 'r') as hdf5_file:
            #         features = hdf5_file['features'][:]
            #     features = torch.from_numpy(features)
            #     torch.save(features, slide_pt_path_new)
        #         print('Saved:', slide_pt_path_new)
        #         break
        #     break
        # break

parser = argparse.ArgumentParser(description='Dataset Preprocess')
parser.add_argument('--feat_dir', type=str, default='R50_features')
args = parser.parse_args()
if __name__ == '__main__':
    project_id = 'iq24'
    # pt_file_regenerate()
    #
    # gen_color_proto(project_id)
    #
    gather_csv(project_id, args.feat_dir)

    # pt_file_regenerate('R50_features')
    # pt_file_regenerate('Color_features')
