"""Test the Drucker's postulate"""

import numpy as np
import os
import torch
import torch.nn as nn
import Training

title = 'BWBN_energy_Drucker'
hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis'
n_samples = 80
Data = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_Processed_data.npz')))
X_train, X_val, y_train, y_val = Data['X_train'], Data['X_val'], Data['y_train'], Data['y_val']
del Data

normalize_gap = 0.1

X = []
y = []

time_length = []
num_inputs = 2 + 1
for i in range(n_samples):
    outputs = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}.npz'.format(i))))
    disp = outputs['disp']
    vel = outputs['vel']
    force = outputs['force']
    energy = outputs['energy']
    X.append(np.concatenate((disp[:, np.newaxis], vel[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
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




for i in range(n_samples):
    f = y[i]
    hysteresis = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, title + '_{}.npz'.format(i))))
    e = hysteresis['energy']
    len_f, len_e = f.shape[0], e.shape[0]
    padding_length = len_f - len_e
    if padding_length > 0:
        e = np.concatenate((e, np.zeros((padding_length))), axis=0)
    f, e = f[np.newaxis, :], e[np.newaxis, :]
    f, e = torch.from_numpy(f).float(), torch.from_numpy(e).float()
    mask = (X[i, :, 2])[np.newaxis, :]
    mask = torch.from_numpy(mask).float()
    print('Drucker\'s postulate loss function for {}-th sample:'.format(i), Training.drucker_loss(f, e, mask).item())
    pass



