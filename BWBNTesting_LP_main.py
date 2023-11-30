import os
import numpy as np
import torch
import torch.nn as nn
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU 환경에서 모든 GPU에 seed 설정
    torch.backends.cudnn.deterministic = True  # cuDNN을 determinstic 모드로 설정
    torch.backends.cudnn.benchmark = False  # cuDNN benchmarking을 off로 설정
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
import EQ_generation
import Preprocessing
import Training
import ResultAnalysis

generate_EQ = False
preprocess = False
train = False
analyze_result = True

Title = 'BWBN_h_energy_diff_disp'


EQ_data_dir = '/home/jaehwan/Python Project/DLCM/Data'
hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis'

processed_data_dir = hysteresis_data_dir
Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
X_max, y_max = Data['X_max'], Data['y_max']

avg_seq_len = int(X_train[:, :, -1].sum()/len(X_train))
protocol_seq_len = 20000
print('Average sequence length:', avg_seq_len)
print('Protocol sequence length set to:', protocol_seq_len)
del Data

periods = [100, 200, 400, 500, 1000, 2000, 4000, 5000]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1, -1, 1/2, 1/4, 1/4]
slopes = [slope / protocol_seq_len * X_max[0] for slope in slopes]
intercepts = [0, 0, 1/2, 0, 1/2]


seed = 0
n_samples = 40
draw_hysteresis = True
mat_type = 'BWBN'
k0 = 6.283**2
mat_props = [0.1, k0, 1., -0.5, 1.5, 1., 0.1, 0.97, 1., 0.2, 0.002, 0.1, 1.0*10**-4, 10**6]
change_at2_to_numpy = False
hysteresis_data_dir_linear_protocol = os.path.normpath(os.path.join(hysteresis_data_dir, './Linear_protocol'))

if generate_EQ:
    EQ_generation.generate_linear_protocol(hysteresis_data_dir_linear_protocol,
                                           draw_hysteresis,
                                           mat_type,
                                           mat_props,
                                           Title,
                                           slopes,
                                           intercepts,
                                           periods,
                                           repetitions)


val_size = 0.2

normalize_gap = 0.1
if preprocess:
    Preprocessing.preprocess_loading_protocol(n_samples,
                                        hysteresis_data_dir_linear_protocol,
                                        processed_data_dir,
                                        title=Title,
                                        val_size=val_size,
                                        normalize_gap=normalize_gap,
                                        X_max=X_max,
                                        y_max=y_max,
                                        test=False)



# Training
Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + 'linear_protocol' + '_Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
del Data
X_train[:, :, 1], X_val[:, :, 1] = np.diff(X_train[:, :, 0], axis=1, prepend=0), np.diff(X_val[:, :, 0], axis=1, prepend=0)
X_train, mask_train, X_val, mask_val = X_train[:, :, :2], X_train[:, :, 2], X_val[:, :, :2], X_val[:, :, 2]
max_len_train = int(np.max(mask_train.sum(axis=-1)))
X_train, mask_train, y_train = X_train[:, :max_len_train, :], mask_train[:, :max_len_train], y_train[:, :max_len_train]
nn_size = 64
alpha = 0.2
num_epochs = 1000
model_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Models'
os.makedirs(model_dir, exist_ok=True) 
window_size = 512
checkpoint_epoch = 5
pretrained = False
checkpoint = False
if pretrained != False:
    checkpoint = torch.load(pretrained)

if train:
    model = Training.train(X_train,
                                y_train,
                                mask_train,
                                X_val,
                                y_val,
                                mask_val,
                                num_epochs,
                                nn_size,
                                alpha,
                                window_size,
                                checkpoint_dir=model_dir,
                                title=Title + '_linear_protocol',
                                checkpoint_epoch=checkpoint_epoch,
                                existing_checkpoint=checkpoint)

Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_val, X_test, y_val, y_test = Data['X_val'], Data['X_test'], Data['y_val'], Data['y_test']
del Data
X_val[:, :, 1] = np.diff(X_val[:, :, 0], axis=1, prepend=0)
X_val, mask_val, X_test, mask_test = X_val[:, :, :2], X_val[:, :, 2], X_test[:, :, :2], X_test[:, :, 2]
# model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title + '_linear_protocol' + '_checkpoint_{}.pth'.format(i+checkpoint_epoch))) for i in range(0, num_epochs, checkpoint_epoch)]
model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title + '_linear_protocol' + '_checkpoint_925.pth'))]
result_plot_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Result Plots'
os.makedirs(result_plot_dir, exist_ok=True)

if analyze_result:
    ResultAnalysis.result_plot(X_val,
                y_val,
                mask_val,
                X_test,
                y_test,
                mask_test,
                nn_size,
                model_paths,
                Title + '_linear_protocol',
                result_plot_dir)
    

