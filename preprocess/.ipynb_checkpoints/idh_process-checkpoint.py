import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def gen_csv(slide_dict, out_root, name):
    df_slide = pd.DataFrame.from_dict(slide_dict)
    df_slide.to_csv(out_root + '/%s'%name, index=False)

def get_stain_idx(stain_proto, rep_batch):
    # print('size check ', stain_proto.size(), rep_batch.size())
    rep_batch_sum = (rep_batch ** 2).sum(dim=-1, keepdims=True)  # [N, 1]
    prototype_gather_sum = (stain_proto ** 2).sum(dim=-1, keepdims=True).T  # [1, M]            #### TODO ####
    distance_matrix = torch.sqrt(
        rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, stain_proto.T))  # [N, M]
    indices = torch.argmin(distance_matrix, dim=-1)
    return indices

def generate_slide_csv(data_root):
    Stages = ['Train', 'Test', 'Valid']
    train_slide_patch = {'slide_id': [], 'patient_id':[], 'patch_id':[], 'label': []}
    test_slide_patch = {'slide_id': [], 'patient_id':[], 'patch_id':[], 'label': []}
    valid_slide_patch = {'slide_id': [], 'patient_id':[], 'patch_id':[], 'label': []}
    error = []
    for st in Stages:
        stage_folder = f'{data_root}/_{st}_All'
        for pat in os.listdir(stage_folder):
            pat_folder = f'{stage_folder}/{pat}'
            pat_label = pat.split('_')[0]
            for patch in os.listdir(pat_folder):
                slide_name = patch.split('.')[0]
                slide_end = patch.split('.')[-1]
                if slide_end != 'jpg':
                    error.append(patch)
                    continue
                if st == 'Train':
                    train_slide_patch['slide_id'].append(slide_name)
                    train_slide_patch['patient_id'].append(pat)
                    train_slide_patch['label'].append(pat_label)
                    train_slide_patch['patch_id'].append(patch)
                elif st == 'Test':
                    test_slide_patch['slide_id'].append(slide_name)
                    test_slide_patch['patient_id'].append(pat)
                    test_slide_patch['label'].append(pat_label)
                    test_slide_patch['patch_id'].append(patch)
                else:
                    valid_slide_patch['slide_id'].append(slide_name)
                    valid_slide_patch['patient_id'].append(pat)
                    valid_slide_patch['label'].append(pat_label)
                    valid_slide_patch['patch_id'].append(patch)
    print('====Train===')
    print('Number of train slides:', len(list(set(train_slide_patch['slide_id']))))
    print('Number of WT patches:', train_slide_patch['label'].count('WT'))
    print('Number of MU patches:', train_slide_patch['label'].count('MU'))
    print('====Test===')
    print('Number of test slides:', len(list(set(test_slide_patch['slide_id']))))
    print('Number of WT slides:', test_slide_patch['label'].count('WT'))
    print('Number of MU slides:', test_slide_patch['label'].count('MU'))
    print('====Valid===')
    print('Number of valid slides:', len(list(set(valid_slide_patch['slide_id']))))
    print('Number of WT slides:', valid_slide_patch['label'].count('WT'))
    print('Number of MU slides:', valid_slide_patch['label'].count('MU'))
    # print('Error:', error)
    # print(list(set(test_slide_patch['slide_id'])))
    gen_csv(train_slide_patch, data_root, 'train_patch.csv')
    gen_csv(test_slide_patch, data_root, 'test_patch.csv')
    gen_csv(valid_slide_patch, data_root, 'valid_patch.csv')

def get_slide_from_patch(patch_df):
    slide_list = patch_df['slide_id'].unique().tolist()
    slide_label = {'slide_id':[], 'label':[]}
    for slide in slide_list:
        slide_label['slide_id'].append(slide)
        slide_label['label'].append(patch_df[patch_df['slide_id'] == slide]['label'].values[0])
    return slide_label
def gather_csv(model_feature):
    local_root_stain_proto = 'data/pre_extracted_color_feature/IDH/Train/'
    SKIP = ['.DS_Store', '._.DS_Store']
    # csv_root = os.getenv('PBS_JOBFS') + '/IDH/'
    csv_root = '/g/data/iq24/IDH/'
    local_root = '/g/data/iq24/IDH'
    save_stain = '/g/data/iq24/IDH'

    if not os.path.exists(save_stain):
        os.makedirs(save_stain)
    stain_proto = torch.load('%s/prototype.pt'%local_root_stain_proto)
    print('Loaded stain prototype:', stain_proto.size())
    csv_dict = {'Train': 'train_patch.csv', 'Test': 'test_patch.csv', 'Valid': 'valid_patch.csv'}
    train_slide = {'slide_id': [], 'label': []}
    test_slide = {'slide_id': [], 'label': []}
    valid_slide = {'slide_id': [], 'label': []}
    for stage in ['Train', 'Test', 'Valid']:
        feature_root = f'/g/data/iq24/IDH/{stage}/{model_feature}/pt_files/'
        patch_list = pd.read_csv(csv_root + csv_dict[stage])
        slide_list = get_slide_from_patch(patch_list)
        for i in tqdm(range(len(slide_list['slide_id']))):
            # slide = slide_list.loc[i]
            # slide_path = slide['slide_id']
            # slide_label = slide['label']
            # slide_name = slide_path.split('/')[-1].split('.')[0]
            slide_name = slide_list['slide_id'][i]
            slide_label = slide_list['label'][i]

            # save color feature
            # stain_path = f'{local_root}/{stage}/pre_extracted_color_feature/pt_files/{slide_name}.pt'
            # # stain_path = '%s/%s/pt_files/'%(local_root, stage) + slide_name+'.pt'
            # stain_ft = torch.load(stain_path)
            # # print(stain_ft.size(), stain_path)
            # if len(stain_ft.size()) > 2:
            #     b, n, _ = stain_ft.size()
            #     stain_ft = stain_ft.view(-1, stain_ft.size(-1))
            # indices = get_stain_idx(stain_proto, stain_ft)
            # stain_path_save = '%s/%s/pre_extracted_color_label/pt_files/'%(save_stain, stage)
            # if not os.path.exists(stain_path_save):
            #     os.makedirs(stain_path_save)
            # torch.save(indices, stain_path_save + slide_name + '.pt')

            # save model feature
            slide_path = f'{feature_root}/{slide_name}.pt'
            if stage == 'Train':
                train_slide['slide_id'].append(slide_path)
                train_slide['label'].append(slide_label)
            elif stage == 'Test':
                test_slide['slide_id'].append(slide_path)
                test_slide['label'].append(slide_label)
            else:
                valid_slide['slide_id'].append(slide_path)
                valid_slide['label'].append(slide_label)

    print('====Train===')
    print('Number of train slides:', len(train_slide['slide_id']))
    print('Number of WT slides:', train_slide['label'].count('WT'))
    print('Number of MU slides:', train_slide['label'].count('MU'))
    print('====Test===')
    print('Number of test slides:', len(test_slide['slide_id']))
    print('Number of WT slides:', test_slide['label'].count('WT'))
    print('Number of MU slides:', test_slide['label'].count('MU'))
    print('====Valid===')
    print('Number of valid slides:', len(valid_slide['slide_id']))
    print('Number of WT slides:', valid_slide['label'].count('WT'))
    print('Number of MU slides:', valid_slide['label'].count('MU'))
    gen_csv(train_slide, f'{csv_root}/Train/{model_feature}', 'train_slide.csv')
    gen_csv(test_slide, f'{csv_root}/Test/{model_feature}', 'test_slide.csv')
    gen_csv(valid_slide, f'{csv_root}/Valid/{model_feature}', 'valid_slide.csv')

parser = argparse.ArgumentParser(description='Dataset Preprocess')
parser.add_argument('--feat_dir', type=str, default='R50_features')
args = parser.parse_args()
if __name__ == '__main__':
    # gather_csv(args.feat_dir)
    generate_slide_csv('/g/data/iq24/IDH/')
    # gather_csv(args.feat_dir)