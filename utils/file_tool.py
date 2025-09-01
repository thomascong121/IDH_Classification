import h5py
import glob
import os
import pandas as pd

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def generate_csv(args):
    all_slide_csv = glob.glob(args.csv_path + '/*_patch.csv')
    print(all_slide_csv)
    all_slide_content = {'patch_id':[]}
    for slide_csv in all_slide_csv:
        slide_content = pd.read_csv(slide_csv)
        for patch_base in slide_content['patch_id']:
            if 'train' in slide_csv:
                patch_root = os.path.join(args.csv_path, '_Train_All', patch_base)
            elif 'test' in slide_csv:
                patch_root = os.path.join(args.csv_path, '_Test_All', patch_base)
            else:
                raise NotImplementedError
            all_slide_content['patch_id'].append(patch_root)
    df = pd.DataFrame.from_dict(all_slide_content)
    print(df.head(10))
    df.to_csv(args.csv_path + '/all_patches.csv', index=False)
    print('Done, output csv: ', args.csv_path + '/all_patches.csv')