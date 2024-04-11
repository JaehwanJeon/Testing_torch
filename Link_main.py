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
import Preprocessing
import Training
import ResultAnalysis

preprocess = True
train = True
analyze_result = True

Title = 'Link_filtered'
file_names = ['Link_' + str(i) + '_filtered' for i in range(1, 5)]

hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis/Pedram'
hysteresis_data_paths = [os.path.normpath(os.path.join(hysteresis_data_dir, './' + file_name + '.csv')) for file_name in file_names]

seed = 0
n_samples = len(file_names)
draw_hysteresis = True

processed_data_dir = hysteresis_data_dir

normalize_gap = 0.1
num_inputs = 2

if preprocess:
    # Total energy as input
    X = []
    y = []

    time_length = []
    for i in range(n_samples):
        outputs = np.genfromtxt(hysteresis_data_paths[i], delimiter=',')
        disp = outputs[:, 0]
        force = outputs[:, 1]
        X.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)
    for i in range(n_samples):
        if len(X[i]) < max_time_length:
            X[i] = np.concatenate((X[i], np.zeros((max_time_length - len(X[i]), num_inputs))), axis=0)
            y[i] = np.concatenate((y[i], np.zeros((max_time_length - len(y[i])))), axis=0)
    X, y = np.array(X), np.array(y)

    if normalize_gap != False:
        X_max = np.max(np.abs(X), axis=(0,1))[:num_inputs-1]
        y_max = np.max(np.abs(y))
        X[:, :, :num_inputs-1] = X[:, :, :num_inputs-1] / (X_max * (1 + normalize_gap))
        y = y / (y_max * (1 + normalize_gap))


    X_train, X_val, y_train, y_val = X[:n_samples-1], X[n_samples-1:], y[:n_samples-1], y[n_samples-1:]  

    np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')), 
             X_train=X_train, X_val=X_val, X_test=X_val,
                y_train=y_train, y_val=y_val, y_test=y_val)

# Training
Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
del Data
X_train, mask_train, X_val, mask_val = X_train[:, :, :num_inputs-1], X_train[:, :, num_inputs-1], X_val[:, :, :num_inputs-1], X_val[:, :, num_inputs-1]
nn_size = 128
alpha = 0.5
num_epochs = 3000
model_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Models'
os.makedirs(model_dir, exist_ok=True) 
window_size = 512
checkpoint_epoch = 5
pretrained = False
checkpoint = False
if pretrained != False:
    checkpoint = torch.load(pretrained)

augmentation_rate = 0.8
result_plot_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Result Plots'

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
X_val, X_test, y_val, y_test = Data['X_val'], Data['X_val'], Data['y_val'], Data['y_val']
del Data
X_val, mask_val, X_test, mask_test = X_val[:, :, :num_inputs-1], X_val[:, :, num_inputs-1], X_test[:, :, :num_inputs-1], X_test[:, :, num_inputs-1]
X_val, y_val = np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0)
mask_val = np.concatenate((mask_train, mask_val), axis=0)
model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title + '_checkpoint_{}.pth'.format(i+checkpoint_epoch))) for i in range(0, num_epochs, checkpoint_epoch)]
# model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_8.pth'))]
os.makedirs(result_plot_dir, exist_ok=True)

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
