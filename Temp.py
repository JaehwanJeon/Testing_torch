"""Test the Drucker's postulate"""

import numpy as np
import os
import torch
import torch.nn as nn
import Training
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
    
    num_samples, time_length = f.size()
    num_chops = 201
    chop_vector = torch.linspace(-1, 1, num_chops).to(f.device)

    # Expand the mask and tensors along the second axis
    mask_expanded = mask.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
    f_expanded = f.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
    e_expanded = e.unsqueeze(1).expand(-1, chop_vector.size(0), -1)

    chop_vector_expanded = chop_vector.unsqueeze(0).unsqueeze(-1).expand(num_samples, -1, time_length)
    deducted_f_expanded = f_expanded - chop_vector_expanded

    # Create deducted_f_sign tensor with size [num_samples, num_chops, time_length]
    deducted_f_sign = (deducted_f_expanded > 0).int()

    diff_sign = torch.diff(deducted_f_sign, dim=2, prepend=deducted_f_sign[:, :, 0].unsqueeze(2))
    change_bool = ((diff_sign != 0) * mask_expanded).bool()
    change_idx = torch.nonzero(change_bool)
    selected_e = e_expanded[change_idx[:, 0], change_idx[:, 1], change_idx[:, 2]]
    diff_e = selected_e[1:] - selected_e[:-1]
    diff_idx = change_idx[1:] - change_idx[:-1]
    diff_e = diff_e * (diff_idx[:, 0] == 0) * (diff_idx[:, 1] == 0)
    diff_e_neg = diff_e[diff_e < 0]

    idx = np.arange(time_length)
    node_1 = change_idx[torch.cat((diff_e, torch.tensor([0.0]))) < 0]
    node_2 = change_idx[torch.cat((torch.tensor([0.0]), diff_e)) < 0]

    fig, ax = plt.subplots()
    ax.plot(idx[mask[0, :].bool()], f[mask.bool()].numpy(), label='Force')
    ax.plot(torch.concat((node_1[:, -1], node_2[:, -1]), dim=0), f[0, torch.concat((node_1[:, -1], node_2[:, -1]), dim=0)].numpy(), 'o', label='Violation')
    ax.set_xlabel('Step')
    ax.set_ylabel('Force (N)')
    ax.legend()
    plt.savefig(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}_Violate_force.png'.format(i))))
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(idx[mask[0, :].bool()], e[mask.bool()].numpy(), label='Energy')
    ax.plot(torch.concat((node_1[:, -1], node_2[:, -1]), dim=0), e[0, torch.concat((node_1[:, -1], node_2[:, -1]), dim=0)].numpy(), 'o', label='Violation')
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy (J)')
    ax.legend()
    plt.savefig(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}_Violate_energy.png'.format(i))))
    plt.close()
    print('Drucker\'s postulate loss function for {}-th sample:'.format(i), Training.drucker_loss(f, e, mask).item())
    pass



