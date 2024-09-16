import pickle
import os
import numpy as np
import nibabel as nib
from utils import Parser

args = Parser()
modalities = ('flair', 't1ce', 't1', 't2')


train_set = {
        'root': '/root/autodl-tmp/TDPC-Net-master/Data/BraTS2019/Valid',
        'flist': 'valid.txt',
        'has_label': True
        }

#        'root': '/hy-tmp/TDPC-Net-master/Data/BraTS2019/Test',
#        'flist': 'test.txt',
#        }

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, cannot find the file!')
        return None
    proxy = nib.load(file_name)
    data = np.asanyarray(proxy.dataobj)
    proxy.uncache()
    return data



def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    if has_label:
        label_file = path + 'seg.nii.gz'
        if not os.path.exists(label_file):
            # Generate an empty label array
            label = np.zeros((240, 240, 155), dtype='uint8', order='C')
        else:
            label = np.array(nib_load(label_file), dtype='uint8', order='C')
    else:
        label = None

    images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]

    output = path + 'data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(4):
        x = images[..., k] #
        y = x[mask] #
        
        lower = np.percentile(y, 0.2) # 算分位数
        upper = np.percentile(y, 99.8)
        
        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]

        x -= y.mean()
        x /= y.std()

        images[..., k] = x
    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return

def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]

    for path in paths:

        process_f32b0(path, has_label)



if __name__ == '__main__':
    doit(train_set)
