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


with open('EQ_list.txt', 'r') as f:
    EQ_list_ = [line.strip() for line in f]
print(EQ_list_[:3])

EQ_list = []
for i, EQ in enumerate(EQ_list_):
    EQ_name = './'+EQ
    EQ_list.append(os.path.normpath(os.path.join('/home/jaehwan/Python Project/DLCM/Data', EQ_name)).replace("\\", "/"))
print(EQ_list[:3])


target_data = os.path.join(EQ_data_dir, './elcentro_NS.txt')
target_data = os.path.normpath(target_data)
seed = 0
gm_scale_factor = 3
n_samples = 80
draw_hysteresis = True
mat_type = 'BWBN'
k0 = 6.283**2
mat_props = [0.1, k0, 1., -0.5, 1.5, 1., 0.1, 0.97, 1., 0.2, 0.002, 0.1, 1.0*10**-4, 10**6]
change_at2_to_numpy = False
impact_length_list = [500, 1000]
impact_magnitude_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
if generate_EQ:
    EQ_generation.generate_hysteresis(EQ_data_dir,
                                                  hysteresis_data_dir,
                                                  target_data,
                                                  change_at2_to_numpy,
                                                  seed,
                                                  gm_scale_factor,
                                                  n_samples,
                                                  draw_hysteresis,
                                                  mat_type,
                                                  mat_props,
                                                  Title,
                                                  EQ_list)
    EQ_generation.generate_impact_response(impact_length_list,
                                            impact_magnitude_list,
                                            hysteresis_data_dir,
                                            draw_hysteresis,
                                            mat_type,
                                            mat_props,
                                            Title)

processed_data_dir = hysteresis_data_dir
val_size = 0.2

normalize_gap = 0.1
if preprocess:
    Preprocessing.preprocess(n_samples,
                                        hysteresis_data_dir,
                                        processed_data_dir,
                                        title=Title,
                                        val_size=val_size,
                                        normalize_gap=normalize_gap)
    Preprocessing.preprocess_impact_loading(impact_length_list,
                                            impact_magnitude_list,
                                            hysteresis_data_dir,
                                            processed_data_dir,
                                            Title)



# Training
Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
del Data
X_train[:, :, 1], X_val[:, :, 1] = np.diff(X_train[:, :, 0], axis=1, prepend=0), np.diff(X_val[:, :, 0], axis=1, prepend=0)
X_train, mask_train, X_val, mask_val = X_train[:, :, :2], X_train[:, :, 2], X_val[:, :, :2], X_val[:, :, 2]
max_len_train = np.argmin(mask_train.sum(axis=0))
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
                                title=Title,
                                checkpoint_epoch=checkpoint_epoch,
                                existing_checkpoint=checkpoint)

Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + 'linear_protocol' + '_Processed_data.npz')))
X_val, X_test, y_val, y_test = Data['X_val'], Data['X_val'], Data['y_val'], Data['y_val']
del Data
X_val[:, :, 1] = np.diff(X_val[:, :, 0], axis=1, prepend=0)
X_val, mask_val, X_test, mask_test = X_val[:, :, :2], X_val[:, :, 2], X_test[:, :, :2], X_test[:, :, 2]
model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title + '_checkpoint_{}.pth'.format(i+checkpoint_epoch))) for i in range(0, num_epochs, checkpoint_epoch)]
# model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_8.pth'))]
result_plot_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Result Plots'
os.makedirs(result_plot_dir, exist_ok=True)

processed_data_dir_impact = os.path.normpath(os.path.join(processed_data_dir, './Impact'))
Data_impact = np.load(os.path.normpath(os.path.join(processed_data_dir_impact, './' + Title + '_Processed_data_impact.npz')))
X_test_impact, y_test_impact = Data_impact['X_test'], Data_impact['y_test']
X_test_impact[:, :, 1] = np.diff(X_test_impact[:, :, 0], axis=1, prepend=0)
del Data_impact
X_test_impact, mask_test_impact = X_test_impact[:, :, :2], X_test_impact[:, :, 2]

if analyze_result:
    ResultAnalysis.result_plot(X_val,
                y_val,
                mask_val,
                X_test,
                y_test,
                mask_test,
                nn_size,
                model_paths,
                Title,
                result_plot_dir)
    
    ResultAnalysis.test_impact(X_test_impact,
                               y_test_impact,
                                mask_test_impact,
                                impact_length_list,
                                impact_magnitude_list,
                                model_paths,
                                Title,
                                nn_size,
                                result_plot_dir)
    

