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
import BWBNTesting_EQ_generation
import BWBNTesting_Preprocessing
import BWBNTesting_Training
# import BWBNTesting_ResultAnalysis_ as BWBNTesting_ResultAnalysis

EQ_generation = False
Preprocessing = False
Training = True
ResultAnalysis = False


EQ_data_dir = '/home/jaehwan/Python Project/DLCM/Data'
hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/BWBN Testing_torch/Hysteresis'


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

if EQ_generation:
    BWBNTesting_EQ_generation.generate_hysteresis(EQ_data_dir,
                                                  hysteresis_data_dir,
                                                  target_data,
                                                  change_at2_to_numpy,
                                                  seed,
                                                  gm_scale_factor,
                                                  n_samples,
                                                  draw_hysteresis,
                                                  mat_type,
                                                  mat_props,
                                                  EQ_list)

processed_data_dir = hysteresis_data_dir
val_size = 0.2

normalize_gap = 0.1
if Preprocessing:
    BWBNTesting_Preprocessing.preprocess(n_samples,
                                        hysteresis_data_dir,
                                        processed_data_dir,
                                        val_size=val_size,
                                        normalize_gap=normalize_gap)



# Training
Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
del Data
X_train, mask_train, X_val, mask_val = X_train[:, :, :2], X_train[:, :, 2], X_val[:, :, :2], X_val[:, :, 2]
nn_size = 40
num_epochs = 300
model_dir = '/home/jaehwan/Python Project/DLCM/BWBN Testing_torch/Models'
os.makedirs(model_dir, exist_ok=True) 
window_size = 200
checkpoint_epoch = 2

if Training:
    model = BWBNTesting_Training.train(X_train,
                                y_train,
                                mask_train,
                                X_val,
                                y_val,
                                mask_val,
                                num_epochs,
                                nn_size,
                                window_size,
                                checkpoint_dir=model_dir,
                                checkpoint_epoch=checkpoint_epoch)

# result_plot_dir = '/home/jaehwan/Python Project/DLCM/BWBN Testing/ResultAnalysis'
# show_plot = True
# range_plot = [40, 80]
# model_name = 'PINN_{}_{}_{}'.format(num_inputs, nn_size, 40)
# model_path = os.path.normpath(os.path.join(model_dir, model_name))

# if ResultAnalysis:
#     BWBNTesting_ResultAnalysis.result_plot(range_plot,
#                                            result_plot_dir,
#                                            show_plot,
#                                            model_path,
#                                            hysteresis_data_dir,
#                                            num_inputs,
#                                            output_factor)
