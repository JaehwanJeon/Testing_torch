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

generate_EQ = True
preprocess = False
train = False
analyze_result = False

Title = 'IMK_PE_PI_DA_2L'


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
mat_type = 'Bilin'

mat_scale_factor = 0.04


E, Fy, Ix, Zx, H, L, d, tw, bf, tf, Lb, ry, My_mod = 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800
E, Fy, Ix, Zx, H, L, d, tw, bf, tf, Lb, ry, My_mod = E * mat_scale_factor, Fy * mat_scale_factor, Ix * mat_scale_factor, Zx * mat_scale_factor, H * mat_scale_factor, L * mat_scale_factor, d * mat_scale_factor, tw * mat_scale_factor, bf * mat_scale_factor, tf * mat_scale_factor, Lb * mat_scale_factor, ry * mat_scale_factor, My_mod * mat_scale_factor 

n = 10.0
c1 = 25.4
c2 = 6.895
K = (n + 1) * 6 * E * Ix / H

theta_p = 0.318 * (d/tw)**-0.550 * (bf/2/tf)**-0.345 * (Lb/ry)**-0.023 * (L/d)**0.090 * (c1 * d/533)**-0.330 * (c2 * Fy/355)**-0.130
theta_pc = 7.5 * (d/tw)**-0.610 * (bf/2/tf)**-0.710 * (Lb/ry)**-0.110 * (c1 * d/533)**-0.161 * (c2 * Fy/355)**-0.320
Lmda = 536 * (d/tw)**-1.260 * (bf/2/tf)**-0.525 * (Lb/ry)**-0.130 * (c2 * Fy/355)**-0.291 / 1000

theta_u = 0.2
Res = 0.4
McMy = 1.1

My_P = My_mod
My_N = -1.0 * My_mod

as_mem_p = (McMy - 1) * My_P / (theta_p * 6 * E * Ix / H)
as_mem_n = -(McMy - 1) * My_N / (theta_p * 6 * E * Ix / H)
SH_mod_p = as_mem_p / (1.0 + n * (1.0 - as_mem_p))
SH_mod_n = as_mem_n / (1.0 + n * (1.0 - as_mem_n))

Lmda = Lmda / 2.0
L_S = Lmda
L_C = Lmda
L_A = Lmda
L_K = Lmda
c_S = 1.0
c_C = 1.0
c_A = 1.0
c_K = 1.0
D_P = 1.0
D_N = 1.0

mat_props = [K, SH_mod_p * 10, SH_mod_n * 10, My_P, My_N, L_S, L_C, L_A, L_K, c_S, c_C, c_A, c_K, theta_p, theta_p, theta_pc, theta_pc, Res, Res, theta_u, theta_u, D_P, D_N]

change_at2_to_numpy = True

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



# Training
Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
del Data
X_train, mask_train, X_val, mask_val = X_train[:, :, :1], X_train[:, :, 1], X_val[:, :, :1], X_val[:, :, 1]
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
augmentation_rate = 0.8

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
                                num_epochs,
                                nn_size,
                                alpha,
                                window_size,
                                checkpoint_dir=model_dir,
                                title=Title,
                                checkpoint_epoch=checkpoint_epoch,
                                existing_checkpoint=checkpoint,
                                augmentation_rate=augmentation_rate,
                                result_plot_dir=result_plot_dir)

Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_val, X_test, y_val, y_test = Data['X_val'], Data['X_test'], Data['y_val'], Data['y_test']
del Data
X_val, mask_val, X_test, mask_test = X_val[:, :, :1], X_val[:, :, 1], X_test[:, :, :1], X_test[:, :, 1]
model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title + '_checkpoint_{}.pth'.format(i+checkpoint_epoch))) for i in range(0, num_epochs, checkpoint_epoch)]
# model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_8.pth'))]

if analyze_result:
    ResultAnalysis.result_plot(X_val,
                y_val,
                mask_val,
                X_test,
                y_test,
                mask_test,
                nn_size,
                model_dir,
                model_paths,
                Title,
                result_plot_dir)
