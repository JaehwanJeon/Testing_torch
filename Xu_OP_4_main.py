import os
import numpy as np
import torch
import torch.nn as nn
import random
from backend import *
import scipy.io

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU 환경에서 모든 GPU에 seed 설정
    torch.backends.cudnn.deterministic = True  # cuDNN을 determinstic 모드로 설정
    torch.backends.cudnn.benchmark = False  # cuDNN benchmarking을 off로 설정
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
import Training
import ResultAnalysis

train = True
analyze_result = True

Title = 'OP_4_PE'


hysteresis_data_dir = 'Xu_hysteresis'


with open('EQ_list.txt', 'r') as f:
    EQ_list_ = [line.strip() for line in f]
print(EQ_list_[:3])

EQ_list = []
for i, EQ in enumerate(EQ_list_):
    EQ_name = './'+EQ
    EQ_list.append(os.path.normpath(os.path.join('/home/jaehwan/Python Project/DLCM/Data', EQ_name)).replace("\\", "/"))
print(EQ_list[:3])


seed = 0

hysteresis_path = os.path.normpath(os.path.join(hysteresis_data_dir, './' + 'data_OP_4_final.mat'))
Data = scipy.io.loadmat(hysteresis_path)
X_train, y_train = Data['X_train'], Data['y_train']
X_train, y_train = X_train[:1000], y_train[:1000]
X_val, y_val = Data['X_valid'], Data['y_valid']
X_test, y_test = Data['X_test'], Data['y_test']
processed_data_dir = hysteresis_data_dir
# X_train_diff, X_val_diff, X_test_diff = np.diff(X_train, axis=1, prepend=0), np.diff(X_val, axis=1, prepend=0), np.diff(X_test, axis=1, prepend=0)
# X_train, X_val, X_test = np.concatenate([X_train, X_train_diff], axis=2), np.concatenate([X_val, X_val_diff], axis=2), np.concatenate([X_test, X_test_diff], axis=2)
y_train, y_val, y_test = y_train[:, :, 0], y_val[:, :, 0], y_test[:, :, 0]
mask_train, mask_val, mask_test = np.ones_like(y_train), np.ones_like(y_val), np.ones_like(y_test)

# Training
device = torch.device('cuda:0')
nn_size = 64
model = CustomLSTM(2, nn_size, 1)
alpha = 0.0
num_epochs = 1000
model_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Models'
os.makedirs(model_dir, exist_ok=True) 
window_size = 1000
checkpoint_epoch = 5
pretrained = False
checkpoint = False
augmentation_rate = False
if pretrained != False:
    checkpoint = torch.load(pretrained)


result_plot_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Result Plots'
os.makedirs(result_plot_dir, exist_ok=True)

if train:
    model = Training.train(X_train,
                                y_train,
                                mask_train,
                                X_val,
                                y_val,
                                mask_val,
                                model,
                                num_epochs,
                                alpha,
                                window_size,
                                checkpoint_dir=model_dir,
                                title=Title,
                                checkpoint_epoch=checkpoint_epoch,
                                existing_checkpoint=checkpoint,
                                augmentation_rate=augmentation_rate,
                                result_plot_dir=result_plot_dir,
                                batch_size=256,
                                device=device)

model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title + '_checkpoint_{}.pth'.format(i+checkpoint_epoch))) for i in range(0, num_epochs, checkpoint_epoch)]

if analyze_result:
    ResultAnalysis.result_plot(X_val,
                y_val,
                mask_val,
                X_test,
                y_test,
                mask_test,
                model,
                model_dir,
                model_paths,
                Title,
                result_plot_dir,
                device = device)
