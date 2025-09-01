import argparse
import os
import cv2
import pandas as pd
import shutil
import torch
from tqdm import tqdm

def get_stain_idx(stain_proto, rep_batch):
    rep_batch_sum = (rep_batch ** 2).sum(dim=-1, keepdims=True)  # [N, 1]
    prototype_gather_sum = (stain_proto ** 2).sum(dim=-1, keepdims=True).T  # [1, M]            #### TODO ####
    distance_matrix = torch.sqrt(
        rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, stain_proto.T))  # [N, M]
    indices = torch.argmin(distance_matrix, dim=-1)
    return indices


def preprocess_df(df, label, target='grade'):
    if label == 'norm':
        df.loc[df.grade == 0, 'grade'] = -1
        df.loc[df.type == 'norm', 'grade'] = 0

    df = df[df[target] >= 0].copy()

    if label != 'both' and label != 'norm':
        df = df[df.type == label].copy()
    cols = df.columns.tolist()
    cols_target = cols.index(target)
    if cols_target != 1:
        old = cols[1]
        cols[1] = target
        cols[cols_target] = old
        df = df[cols]
    return df
def _fill(cur_csv, slide_dict, patch_dict, mag):
    for wsi in cur_csv.wsi.unique():
        if wsi not in slide_dict['slide_id']:
            slide_dict['slide_id'].append(wsi)
            wsi_label = cur_csv[cur_csv.wsi == wsi].top_label_name
            assert len(wsi_label.unique()) == 1
            slide_dict['label'].append(wsi_label.unique()[0])
            wsi_grade = cur_csv[cur_csv.wsi == wsi].grade
            assert len(wsi_grade.unique()) == 1
            slide_dict['grade'].append(wsi_grade.unique()[0])

        wsi_patch = cur_csv[cur_csv.wsi == wsi].image_id.tolist()
        wsi_patch_label = cur_csv[cur_csv.wsi == wsi].top_label_name.tolist()
        wsi_patch_slide = cur_csv[cur_csv.wsi == wsi].wsi.tolist()
        wsi_patch_grade = cur_csv[cur_csv.wsi == wsi].grade.tolist()

        wsi_patch = ['%s/%s/%s' % (mag, wsi_patch_label[i], wsi_patch[i]) for i in range(len(wsi_patch))]
        patch_dict['patch_id'] += wsi_patch
        patch_dict['label'] += wsi_patch_label
        patch_dict['slide_id'] += wsi_patch_slide
        patch_dict['grade'] += wsi_patch_grade
        patch_dict['mag'] += [mag]*len(wsi_patch_grade)
    return slide_dict, patch_dict

def gen_csv(slide_dict, out_root, name):
    df_slide = pd.DataFrame.from_dict(slide_dict)
    df_slide.to_csv(out_root + '/%s'%name, index=False)
    print('CSV generated @ %s'%out_root + '/%s'%name)

def get_csv():
    root = [os.getenv('PBS_JOBFS') + '/unitopath-public/7000',
            os.getenv('PBS_JOBFS') + '/unitopath-public/800']
    out_root = os.getenv('PBS_JOBFS') + '/UNITOPatho'
    os.makedirs(out_root, exist_ok=True)
    train_slide_dict= {'slide_id':[], 'label':[], 'grade':[]}
    train_patch_dict = {'patch_id':[], 'label':[], 'slide':[], 'grade':[], 'mag':[]}
    test_slide_dict= {'slide_id':[], 'label':[], 'grade':[]}
    test_patch_dict = {'patch_id':[], 'label':[], 'slide':[], 'grade':[], 'mag':[]}
    for mag in root:
        mag_value = mag.split('/')[-1]
        for csv in ['train.csv', 'test.csv']:
            cur_csv = pd.read_csv(mag + '/%s'%csv)
            if 'train' in csv:
                train_slide_dict, train_patch_dict = _fill(cur_csv, train_slide_dict, train_patch_dict, mag_value)
            else:
                test_slide_dict, test_patch_dict = _fill(cur_csv, test_slide_dict, test_patch_dict, mag_value)

    print('======Trian=======')
    print('Total WSI ',len(train_slide_dict['slide_id']))
    print('Total patches ', len(train_patch_dict['patch_id']))
    print('======Test=======')
    print('Total WSI ',len(test_slide_dict['slide_id']))
    print('Total patches ', len(test_patch_dict['patch_id']))
    # generate csv
    gen_csv(train_slide_dict, out_root, 'train_slide.csv')
    gen_csv(train_patch_dict, out_root, 'train_patch.csv')
    gen_csv(test_slide_dict, out_root, 'test_slide.csv')
    gen_csv(test_patch_dict, out_root, 'test_patch.csv')

def patchify(csv, size=224):
    img_root = os.getenv('PBS_JOBFS') + '/unitopath-public'
    out_root = os.getenv('PBS_JOBFS') + '/UNITOPatho_test/patches'
    os.makedirs(out_root, exist_ok=True)
    df = pd.read_csv(csv)
    patch_cont = {'image_id':[], 'label':[], 'slide':[], 'grade':[]}
    mag_split = df.mag.unique().tolist()
    for mag in mag_split:
        df_mag = df[df.mag==mag]
        top_label = df_mag.label.unique().tolist()
        for lb in top_label:
            df_lb = df_mag[df_mag.label==lb]
            print('Patching %s/%s'%(mag, lb))
            for i in tqdm(range(len(df_lb))):
                entry = df_lb.iloc[i]
                img_pth = img_root + '/%s'%entry[0]
                slide_id = entry[2]
                img_base = os.path.splitext(img_pth.split('/')[-1])[0]
                if 'train' in csv:
                    out_img = 'train/%s/%s' % (slide_id, img_base)
                else:
                    out_img = 'test/%s/%s' % (slide_id, img_base)
                out_root_img = out_root + '/%s'%out_img
                os.makedirs(out_root_img, exist_ok=True)
                img_np = cv2.imread(img_pth)
                y_len, x_len, _ = img_np.shape
                scale_y = y_len//size
                scale_x = x_len//size
                count = 0
                for y in range(scale_y):
                    y_up = min(y_len, (y + 1) * size)
                    if y_up - y*size < size:
                        offset = y_up - y*size
                        y_low = y*size - offset
                    else:
                        y_low = y*size
                    for x in range(scale_x):
                        x_up = min(x_len, (x + 1) * size)
                        if x_up - x*size < size:
                            offset = x_up - x * size
                            x_low = x * size - offset
                        else:
                            x_low = x * size
                        cropped_img = img_np[y_low:y_up, x_low:x_up]
                        assert cropped_img.shape[0] == cropped_img.shape[1]
                        assert cropped_img.shape[0] == size
                        out_crop = out_root_img + '/%s.jpg'%count
                        patch_cont['image_id'].append(out_img + '/%s.jpg'%count)
                        patch_cont['label'].append(entry[1])
                        patch_cont['slide'].append(entry[2])
                        patch_cont['grade'].append(entry[3])
                        cv2.imwrite(out_crop, cropped_img)
                        count += 1
                        # print(filename)
                # print('%s has %d patches '%(out_root_img, count))

            print('Done patching %s/%s' % (mag, lb))
            print('remove ',img_root + '/%s/%s'%(mag, lb))
            # shutil.rmtree(img_root + '/%s/%s' % (mag, lb))
            shutil.move(img_root + '/%s/%s' % (mag, lb), '/scratch/sz65/cc0395/WSI/')
        df_patch = pd.DataFrame.from_dict(patch_cont)
        df_patch.to_csv(csv, index=False)
        print('Total patch ', len(df_patch))
            # shutil.rmtree(img_root + '/%s/%s'%(mag, lb))

def gather_csv():
    csv_root = os.getenv('PBS_JOBFS')
    local_root = f'{csv_root}/unitopath-public/'
    csv_800 = csv_root + '/unitopath-public/800/'
    csv_7000 = csv_root + '/unitopath-public/7000_224/'
    train_patch = {'patch_id':[], 'label':[], 'slide_id':[], 'grade':[], 'mag':[]}
    test_patch = {'patch_id':[], 'label':[], 'slide_id':[], 'grade':[], 'mag':[]}
    train_slide = {'slide_id': [], 'label': [], 'grade': []}
    test_slide = {'slide_id': [], 'label': [], 'grade': []}
    for stage in ['train.csv', 'test.csv']:
        if stage == 'train.csv':
            df800 = pd.read_csv(csv_800 + stage)
            train_slide, train_patch = _fill(df800, train_slide, train_patch, '800')
            df7000 = pd.read_csv(csv_7000 + stage)
            train_slide, train_patch = _fill(df7000, train_slide, train_patch, '7000_224')
        else:
            df800 = pd.read_csv(csv_800 + stage)
            test_slide, test_patch = _fill(df800, test_slide, test_patch, '800')
            df7000 = pd.read_csv(csv_7000 + stage)
            test_slide, test_patch = _fill(df7000, test_slide, test_patch, '7000_224')
    df_train_patch = pd.DataFrame.from_dict(train_patch)
    df_test_patch = pd.DataFrame.from_dict(test_patch)
    df_train_slide = pd.DataFrame.from_dict(train_slide)
    df_test_slide = pd.DataFrame.from_dict(test_slide)
    print('====Train===')
    print('number of slides ', len(df_train_slide.slide_id.unique()))
    print('number of patches ', len(df_train_patch.patch_id.unique()))
    print('====Test===')
    print('number of slides ', len(df_test_slide.slide_id.unique()))
    print('number of patches ', len(df_test_patch.patch_id.unique()))
    # df_train_slide.to_csv(csv_root+'/unitopath-public/train_slide.csv' , index=False)
    # df_train_patch.to_csv(csv_root+'/unitopath-public/train_patch.csv', index=False)
    # df_test_slide.to_csv(csv_root+'/unitopath-public/test_slide.csv', index=False)
    # df_test_patch.to_csv(csv_root+'/unitopath-public/test_patch.csv', index=False)
    df_train_slide.to_csv(local_root+'/train_slide.csv' , index=False)
    df_train_patch.to_csv(local_root+'/train_patch.csv', index=False)
    df_test_slide.to_csv(local_root+'/test_slide.csv', index=False)
    df_test_patch.to_csv(local_root+'/test_patch.csv', index=False)
    print('Done gathering csv')

def get_info(target, unit='gb'):
    root = os.getenv('PBS_JOBFS')
    train_csv = '%s/unitopath-public/train_patch.csv' % (root)

    train_df = pd.read_csv(train_csv)
    # train_df = preprocess_df(train_df, 'both', target)
    print('\n---- DATA SUMMARY ----')
    print('---------------------------------- Train ----------------------------------')
    print(train_df.groupby(target).count())

    # total patch:
    total = 0
    for i in tqdm(range(len(train_df))):
        img_pth = train_df.iloc[i,0]
        slide_id = train_df.iloc[i,2]
        img_pth = '%s/unitopath-public'%root +'/'+img_pth
        file_size = os.path.getsize(img_pth)
        total += file_size
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    print('total %.2f GB'%(total / 1024 ** exponents_map[unit]))

def re_format_csv(model_feature):
    if 'hipt' in model_feature:
        csv_path = 'VIT_DINO_HIPT_features'
    elif 'ViT' in model_feature:
        csv_path = 'VIT_features'
    elif 'ResNet' in model_feature:
        csv_path = 'R50_features'
    elif 'R50_new' in model_feature:
        csv_path = 'R50_new'
    else:
        raise NotImplementedError
    csv_root = f'/g/data/iq24/unitopath-public/'
    out_path = f'/g/data/iq24/unitopath-public/{csv_path}/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    SKIP = ['.DS_Store', '._.DS_Store']
    local_root = 'data/pre_extracted_color_feature/unitopath-public'
    save_stain = 'data/pre_extracted_color_label/unitopath-public'
    local_root_stain_proto = 'data/pre_extracted_color_feature/unitopath-public/Train/'
    if not os.path.exists(save_stain):
        os.makedirs(save_stain)
    stain_proto = torch.load('%s/prototype.pt'%local_root_stain_proto)
    print('Loaded stain prototype:', stain_proto.size())
    csv_dict = {'Train': 'train_slide.csv', 'Test': 'test_slide.csv'}
    train_slide = {'slide_id': [], 'label': []}
    test_slide = {'slide_id': [], 'label': []}
    for stage in ['Train', 'Test']:
        feature_root = f'data/pre_extracted_feature/unitopath-public/{model_feature}/{stage}/pt_files/'
        slide_list = pd.read_csv(csv_root + csv_dict[stage])
        print(f'Total {stage} slides ', len(slide_list))
        exist = 0
        for i in range(len(slide_list)):
            slide = slide_list.loc[i]
            slide_path = slide['slide_id']
            slide_path = slide_path.split('/')[-1]
            if '.pt.pt' in slide_path:
                slide_path = slide_path.replace('.pt.pt', '.pt')
            slide_label = slide['label']
            # save color feature
            stain_path = '%s/%s/pt_files/'%(local_root, stage) + slide_path
            stain_ft = torch.load(stain_path)
            if len(stain_ft.size()) > 2:
                b, n, _ = stain_ft.size()
                stain_ft = stain_ft.view(-1, stain_ft.size(-1))
            indices = get_stain_idx(stain_proto, stain_ft)
            stain_path_save = '%s/%s/pt_files/'%(save_stain, stage)
            if not os.path.exists(stain_path_save):
                os.makedirs(stain_path_save)
            torch.save(indices, stain_path_save + slide_path)
            # save model feature
            slide_path = f'{feature_root}/{slide_path}'
            if stage == 'Train':
                train_slide['slide_id'].append(slide_path)
                train_slide['label'].append(slide_label)
            else:
                test_slide['slide_id'].append(slide_path)
                test_slide['label'].append(slide_label)
    print('====Train===')
    print('Number of train slides:', len(train_slide['slide_id']))
    print('====Test===')
    print('Number of test slides:', len(test_slide['slide_id']))
    gen_csv(train_slide, out_path, 'train_slide.csv')
    gen_csv(test_slide, out_path, 'test_slide.csv')


parser = argparse.ArgumentParser(description='Dataset Preprocess')
parser.add_argument('--feat_dir', type=str, default='R50_features')
args = parser.parse_args()
if __name__ == '__main__':
    # get_csv()
    # patchify(os.getenv('PBS_JOBFS') + '/UNITOPatho/test_patch.csv')
    # get_info('label')
    # / jobfs / 114385746.
    # gadi - pbs / UNITOPatho / training / patches / train / 800 / TVA.LG / TVA.LG
    # CASO
    # 2 - 2018 - 12 - 04
    # 13.19
    # .16.ndpi_ROI__mpp0
    # .44
    # _reg000_crop_sk00021_(9060, 9947, 1812, 1812).png
    # gather_csv()
    re_format_csv(args.feat_dir)