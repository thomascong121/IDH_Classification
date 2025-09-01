import time

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import h5py
import os
import openslide
import pandas as pd
from tqdm import tqdm


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return trnsfrms_val


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 task,
                 file_path,
                 wsi,
                 label,
                 pretrained=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1,
                 K=10
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms
        self.task = task
        self.file_path = file_path
        self.label_dict = {'IDH': {'WT': 0, 'MU': 1},  # {'normal_tissue': 0, 'tumor_tissue': 1},
                      'CAM16': {'normal': 0, 'tumor': 1},
                      'unitopath-public': {'TVA.LG': 0,
                                           'TA.LG': 1,
                                           'TVA.HG': 2,
                                           'NORM': 3,
                                           'TA.HG': 4,
                                           'HP': 5}}
        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        self.label = int(self.label_dict[self.task][label])
        self.K = K
        # self.summary()

    def __len__(self):
        return min(self.length, self.K)

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, self.label


if __name__ == '__main__':
    format = 'tif'
    data_dir = '/g/data/iq24/CAMELYON16'
    data_root = os.getenv('PBS_JOBFS')
    train_csv = f'{data_root}/CAM16/train_slide.csv'
    test_csv = f'{data_root}/CAM16/test_slide.csv'
    train_df = pd.read_csv(train_csv)
    start = time.time()
    for i in range(len(train_df)):
        slide_id = train_df.iloc[i,0]
        label = train_df.iloc[i,1]
        print(slide_id, label)
        slide_name = slide_id.split('/')[-1].split('.')[0]
        _slide_id_ = slide_id.replace('R50_features', 'patches')
        _slide_id_ = _slide_id_.replace('pt_files', '')
        coord_path = _slide_id_.replace(f'{slide_name}.pt', f'{slide_name}.h5')
        stage_label = '/'.join(slide_id.split('/')[4:6])

        slide_file_path = f'{data_dir}/{stage_label}/{slide_name}.{format}'
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(task='CAM16',
                                     file_path=coord_path, wsi=wsi,
                                     label=label,
                                     pretrained=True,
                                     custom_downsample=1,
                                     target_patch_size=-1)
        loader_kwargs = {'num_workers': 8, 'pin_memory': True}
        loader = DataLoader(dataset=dataset, batch_size=512, **loader_kwargs)
        for count, data in enumerate(tqdm(loader)):
            pass
    print('Finished loading patches in: ', time.time()-start, 's')