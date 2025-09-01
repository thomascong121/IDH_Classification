import os
import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from scipy import stats


def preprocess_df(df, label, target='grade'):
    if label == 'norm':
        df.loc[df.grade == 0, 'grade'] = -1
        df.loc[df.type == 'norm', 'grade'] = 0

    df = df[df[target] >= 0].copy()

    if label != 'both' and label != 'norm':
        df = df[df.type == label].copy()
    return df


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 label_col=None,
                 dataset_name='IDH',
                 test_label='label'
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            patient_strat (boolean): Whether the dataset contains patient info
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.dataset_name = dataset_name
        self.test_label = test_label
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        slide_data = pd.read_csv(csv_path)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        self.slide_data = slide_data
        self.slide_cls_prep()

    def csv_preprocess(self):
        pass

    def slide_pth_preprocess(self):
        pass

    def computeposweight(self):
        pass

    def get_bag_sizes(self):
        pass
    def slide_cls_prep(self):
        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def return_splits(self,
                           csv_path=None,
                           test_folder=False,
                           preprocess=False,
                           num_class=2,
                           print_inf=False,
                           test_label='label',
                           transform = None,
                           balance=False
                      ):
        assert csv_path
        train_split = pd.read_csv(csv_path['train'], dtype=self.slide_data['slide_id'].dtype)
        val_split = pd.read_csv(csv_path['test'], dtype=self.slide_data['slide_id'].dtype)
        if test_label != 'label':
            label_dict = {'0': 0, '1': 1}
            train_split[test_label] = train_split[test_label].map(label_dict, na_action=None)
            val_split[test_label] = val_split[test_label].map(label_dict, na_action=None)
            train_split = preprocess_df(train_split, 'both', test_label)
            val_split = preprocess_df(val_split, 'both', test_label)
            print(train_split.groupby(test_label).count())
            if 'slide' in train_split.columns:
                print(len(train_split.slide.unique()), 'WSIs')
            print(val_split.groupby(test_label).count())
            if 'slide' in val_split.columns:
                print(len(val_split.slide.unique()), 'WSIs')
        train_split = Generic_ImageSplit(train_split, data_dir=self.data_dir,
                                         num_classes=num_class,
                                         label_dict=self.label_dict,
                                         dataset_name=self.dataset_name,
                                         test_label=test_label,
                                         transform=transform,
                                            balance=balance)
        val_split = Generic_ImageSplit(val_split, data_dir=self.data_dir,
                                       num_classes=num_class,
                                       label_dict=self.label_dict,
                                       dataset_name=self.dataset_name,
                                       test_label=test_label,
                                       transform=transform,
                                            balance=balance)
        if test_folder:
            assert isinstance(test_folder, str)
            csv_test = test_folder + '/test.csv'
            test_data = pd.read_csv(csv_test)
            test_split = Generic_ImageSplit(test_data, data_dir=test_folder,
                                            num_classes=num_class,
                                            label_dict=self.label_dict,
                                            dataset_name=self.dataset_name,
                                            test_label=test_label,
                                            transform=transform,
                                            balance=balance)
        else:
            test_split = None
        if preprocess:
            train_split.slide_pth_preprocess(stage='train')
            train_split.csv_preprocess()
            train_split.bag_mu, train_split.bag_max, train_split.bag_min = train_split.get_bag_sizes()
            print(f'Train mu {train_split.bag_mu} | min {train_split.bag_min} | max {train_split.bag_max}\n')
            val_split.slide_pth_preprocess(stage='val')
            val_split.csv_preprocess()
            val_split.bag_mu, val_split.bag_max, val_split.bag_min = val_split.get_bag_sizes()
            print(f'Val mu {val_split.bag_mu} | min {val_split.bag_min} | max {val_split.bag_max}\n')
            if test_split is not None:
                test_split.slide_pth_preprocess(stage='test')
                test_split.csv_preprocess()
                test_split.bag_mu, test_split.bag_max, test_split.bag_min = test_split.get_bag_sizes()
                print(f'Test mu {test_split.bag_mu} | min {test_split.bag_min} | max {test_split.bag_max}\n')
        # if print_inf:
        #     print("Training on %d samples" % (len(train_split)))
        #     print("Validating on {} samples".format(len(val_split)))

            if test_split is not None:
                print("Testing on {} samples".format(len(test_split)))
        if test_split is None:
            return train_split, val_split, []
        return train_split, val_split, test_split


class Generic_MIL_ImageDataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 data_dir,
                 **kwargs):
        super(Generic_MIL_ImageDataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.slide_lookup_dic = None

    def slide_pth_preprocess(self, stage='train'):
        lookup_dict = {}
        prefix_map = {'IDH': ['_Train_All', '_Test_All']}

        for i in range(len(self.slide_data)):
            # slide_pth = self.slide_data.iloc[i][0]
            # slide_pth_list = slide_pth.split('/')
            # slide_id = slide_pth_list[-1]
            # if self.dataset_name in prefix_map:
            #     if stage == 'train':
            #         slide_pth_new = self.data_dir + '/%s/%s' % (prefix_map[self.dataset_name][0], slide_pth)
            #     else:
            #         slide_pth_new = self.data_dir + '/%s/%s' % (prefix_map[self.dataset_name][1], slide_pth)
            # else:
            #     slide_pth_new = os.environ.get('PBS_JOBFS') + '/%s/%s' % (self.dataset_name, slide_pth)
            patch_pth = self.slide_data.iloc[i]['patch_id']
            patient_pth = self.slide_data.iloc[i]['patient_id']
            if stage == 'train':
                slide_pth_new = self.data_dir + '/%s/%s/%s' % (prefix_map[self.dataset_name][0], patient_pth, patch_pth)
            else:
                slide_pth_new = self.data_dir + '/%s/%s/%s' % (prefix_map[self.dataset_name][1], patient_pth, patch_pth)
            lookup_dict[patch_pth] = slide_pth_new
        self.slide_lookup_dic = lookup_dict

    def csv_preprocess(self, fix_number=3):
        for i in range(len(self.slide_data)):
            slide_pth = self.slide_data.iloc[i]['patch_id']
            slide_pth_list = slide_pth.split('/')
            slide_id = slide_pth_list[-1]
            if self.slide_lookup_dic is not None:
                self.slide_data.iloc[i]['patch_id'] = self.slide_lookup_dic[slide_id]
            else:
                # TODO
                continue

    def computeposweight(self):
        pos_count  = 0
        count_dict = {x: 0 for x in range(len(self.label_dict))}
        labels     = []
        for item in range(len(self.bags_list)):
            slide_id = self.bags_list[item]
            bag_rows = self.slide_data[self.slide_data['slide_id'] == slide_id]
            cls_id = bag_rows.iloc[0]['label']  # all patches in a slide share the same label
            if isinstance(cls_id, str):
                cls_id = self.label_dict[cls_id]
            pos_count += cls_id
            count_dict[cls_id] += 1
            labels.append(cls_id)
        return torch.tensor((len(self.slide_data)-pos_count)/pos_count), count_dict, labels

    def get_bag_sizes(self):
        bags = []
        for item in range(len(self.bags_list)):
            slide_id = self.bags_list[item]
            bag_rows = self.slide_data[self.slide_data['slide_id'] == slide_id]
            bags.append(len(bag_rows))
        return np.mean(bags),np.max(bags), np.min(bags)

    def __getitem__(self, idx):
        slide_id = self.bags_list[idx]
        bag_rows = self.slide_data[self.slide_data['slide_id'] == slide_id]
        label = bag_rows.iloc[0]['label']  # all patches in a slide share the same label
        if label in list(self.label_dict.keys()):
            label = self.label_dict[label]
        tiles = []
        for i in range(len(bag_rows)):
            img_path = bag_rows.iloc[i]['patch_id']
            img_pil = Image.open(img_path)
            img_tensor = self.transform(img_pil)
            tiles.append(img_tensor.unsqueeze(0))
        tiles = torch.cat(tiles)
        if self.balance: # TODO: need check
            sizes  = tiles.size()
            n,c,h,w = sizes
            bag_feats = torch.zeros((self.bag_max,c,h,w), dtype=torch.float)
            bag_feats[:n,:] = tiles
            tiles = bag_feats
        return tiles, int(label)


class Generic_ImageSplit(Generic_MIL_ImageDataset):
    def __init__(self, slide_data, data_dir=None, num_classes=2, label_dict={}, dataset_name=None,
                 test_label='label', transform=None, balance=False):
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.label_dict = label_dict
        self.dataset_name = dataset_name
        self.test_label = test_label
        self.bags_list = self.slide_data['slide_id'].unique().tolist()
        self.balance = balance
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        self.transform = transform
        print('total ', len(self.bags_list))
        self.pos_weight, self.count_dict, self.labels = self.computeposweight()

    def __len__(self):
        return len(self.bags_list)
