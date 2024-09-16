

## Requirements
* python 3.8
* pytorch 1.12
* nibabel
* pickle 
* imageio
* pyyaml

## Implementation

Download the BraTS2019 dataset and change the path:

```
experiments/PATH.yaml
```

### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance . 

```
python preprocess.py
```

### Training

Sync bacth normalization is used so that a proper batch size is important to obtain a decent performance. Multiply gpus training with batch_size=10 is recommended.The total training time is about 12 hours and the average prediction time for each volume is 2.3 seconds when using randomly cropped volumes of size 128×128×128 and batch size 10 on two parallel Nvidia Tesla K40 GPUs for 800 epochs.

```
python train_all.py --gpu=0 --cfg=TDPC_Net --batch_size=10
```

### Test

You could obtain the resutls as paper reported by running the following code:

```
python test.py --mode=1 --is_out=True --verbose=True --use_TTA=True --postprocess=True --snapshot=True --restore=model_last.pth --cfg=TDPC_Net --gpu=0
```
PATH.yaml
train_data_dir: /root/autodl-tmp/TDPC-Net-master/Data/BraTS2019/Train
valid_data_dir: /root/autodl-tmp/TDPC-Net-master/Data/BraTS2019/Valid
#test_data_dir: /mip/Data/BraTS2018/Test
ckpt_dir: ./ckpts

TDPC_Net.yaml

net: TDPC_Net
net_params:
  in_dim: 4
  out_dim: 4
  num_filters: 32

criterion: GeneralizedDiceLoss

weight_type: square
alpha : 0.4
gamma: 1.0
eps: 1e-6
dataset: BraTSDataset
seed: 1024
batch_size: 8
num_epochs: 900
save_freq: 10     # save every 50 epochs
valid_freq: 10   # validate every 10 epochs
start_iter: 0

opt: Adam
opt_params:
  lr: 1e-3
  weight_decay: 1e-5
  amsgrad: true

workers: 15

train_list: all.txt
valid_list: valid.txt
train_transforms: # for training
  Compose([
    RandCrop3D((128,128,128)),
    RandomRotion(10), 
    RandomIntensityChange((0.1,0.1)),
    RandomFlip(0),
    NumpyType((np.float32, np.int64)),
    ])
test_transforms: # for testing
  Compose([
    Pad((0, 0, 0, 5, 0)),
    NumpyType((np.float32, np.int64)),
    ])

