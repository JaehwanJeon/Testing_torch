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

preprocess = False
train = True
analyze_result = True

Title = 'WhittakerPeer_novel'
file_name = 'Whittaker-PEER-0007-SpecimenRC01-MomentRotation'

hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis/WhittakerPeer'
hysteresis_data_path = os.path.normpath(os.path.join(hysteresis_data_dir, './' + file_name + '.csv'))

seed = 0
n_samples = 1
draw_hysteresis = True

processed_data_dir = hysteresis_data_dir

normalize_gap = 0.1
if preprocess:
    # Total energy as input
    X = []
    y = []

    time_length = []
    num_inputs = 2 + 1
    for i in range(n_samples):
        outputs = np.genfromtxt(hysteresis_data_path, delimiter=',')
        disp = outputs[:, 0]
        force = outputs[:, 1]
        X.append(np.concatenate((disp[:, np.newaxis], disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
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
        X_max = np.max(np.abs(X), axis=(0,1))[:2]
        y_max = np.max(np.abs(y))
        X[:, :, :2] = X[:, :, :2] / (X_max * (1 + normalize_gap))
        y = y / (y_max * (1 + normalize_gap))


    X_train, y_train = X, y

    np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')), X_train=X_train, X_test=X_train, X_val=X_train, 
             y_train=y_train, y_test=y_train, y_val=y_train)


# Training
Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
del Data
X_train, mask_train, X_val, mask_val = X_train[:, :, :1], X_train[:, :, 2], X_val[:, :, :1], X_val[:, :, 2]
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

Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
X_val, X_test, y_val, y_test = Data['X_val'], Data['X_test'], Data['y_val'], Data['y_test']
del Data
X_val, mask_val, X_test, mask_test = X_val[:, :, :1], X_val[:, :, 2], X_test[:, :, :1], X_test[:, :, 2]
model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title + '_checkpoint_{}.pth'.format(i+checkpoint_epoch))) for i in range(0, num_epochs, checkpoint_epoch)]
# model_paths = [os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_8.pth'))]
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
                Title,
                result_plot_dir)
