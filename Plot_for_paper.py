#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import ReadRecord
from backend import *
import Experiment_230508 as Experiment

#%%

plt.rcParams['font.size'] = 7 # 기본 글씨 크기 설정
plt.rcParams['font.family'] = 'serif' # 글씨체 기본 설정
plt.rcParams['axes.titlesize'] = 7 # 제목의 기본 글씨 크기 설정
plt.rcParams['axes.labelsize'] = 8 # 축 라벨의 기본 글씨 크기 설정
plt.rcParams['xtick.labelsize'] = 7 # x축 눈금 라벨의 기본 글씨 크기 설정
plt.rcParams['ytick.labelsize'] = 7 # y축 눈금 라벨의 기본 글씨 크기 설정
plt.rcParams['legend.fontsize'] = 7 # 범례의 기본 글씨 크기 설정


device = torch.device('cuda:1')
save_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/ResultAnalysis_for_paper/Paper'
os.makedirs(save_dir, exist_ok=True)
generate_data = False
save_data_dir = os.path.join(save_dir, 'Data')
os.makedirs(save_data_dir, exist_ok=True)


EQ_data_dir = '/home/jaehwan/Python Project/DLCM/Data'
hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis'
processed_data_dir = hysteresis_data_dir
model_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Models'
loss_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Result Plots'





#%% Plot example earthquakes
with open('EQ_list.txt', 'r') as f:
    EQ_list_ = [line.strip() for line in f]
print(EQ_list_[:3])

EQ_list = []
for i, EQ in enumerate(EQ_list_):
    EQ_name = './'+EQ
    EQ_list.append(os.path.normpath(os.path.join('/home/jaehwan/Python Project/DLCM/Data', EQ_name)).replace("\\", "/"))
print(EQ_list[:3])

gm_list, t_list = [], []
for EQ_name in EQ_list:
    file_name, file_extension = os.path.splitext(EQ_name)
    dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
    file_name, file_extension = os.path.splitext(EQ_name)
    gm = np.load(file_name+'.npy')
    t = np.arange(0, nPts*dt, dt)
    gm_list.append(gm)
    t_list.append(t)

fig, ax = plt.subplots(3, 1, figsize=(6, 6))
plt.subplots_adjust(top = 0.97, wspace=0.1, hspace=0.3)

ax[0].plot(t_list[0], gm_list[0] / 9.8, color='k', linewidth=0.7)
ax[0].set_xlim([t_list[0][0], t_list[0][-1]])
ax[0].grid(True)
ax[0].text(0.5, -0.15, '(a)', transform=ax[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
ax[1].plot(t_list[1], gm_list[1] / 9.8, color='k', linewidth=0.7)
ax[1].set_xlim([t_list[1][0], t_list[1][-1]])
ax[1].grid(True)
ax[1].text(0.5, -0.15, '(b)', transform=ax[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
ax[1].set_ylabel('Ground accelration [g]', labelpad=7)
ax[2].plot(t_list[-1], gm_list[-1] / 9.8, color='k', linewidth=0.7)
ax[2].set_xlim([t_list[-1][0], t_list[-1][-1]])
ax[2].text(0.5, -0.3, '(c)', transform=ax[2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
ax[2].grid(True)
ax[2].set_xlabel('Time [s]')
fig.savefig(os.path.join(save_dir, '1989LomaPrieta_1994Northridge_1971SanFernando.svg'), bbox_inches='tight')





# %% Plot Bouc-Wen class test results
Titles = ['BoucWen_PE_PI_DA', 'BWBN_PE_PI_DA']
# Load best model
for Title in Titles:
    if generate_data:
        loss_data = pd.read_csv(os.path.normpath(os.path.join(loss_dir, Title + '_loss.csv')))
        losses = loss_data['Loss'].values

        best_model_idx = np.argmin(losses)
        best_model_epoch = loss_data['Epoch'].values[best_model_idx]
        checkpoint = torch.load(os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_{}.pth'.format(int(best_model_epoch)))))

        nn_size = checkpoint['cell.fc_f.bias'].size()[0]
        model = CustomLSTM(2, nn_size, 1)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        # Load test data
        Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
        X_val, X_test, y_val, y_test = Data['X_val'], Data['X_test'], Data['y_val'], Data['y_test']
        del Data
        X_val, mask_val, X_test, mask_test = X_val[:, :, :1], X_val[:, :, 1], X_test[:, :, :1], X_test[:, :, 1]
        X_val, X_test = torch.from_numpy(X_val).float().to(device), torch.from_numpy(X_test).float().to(device)
        X_val, X_test = add_diff(X_val), add_diff(X_test)
        y_val, y_test = torch.from_numpy(y_val).float().to(device), torch.from_numpy(y_test).float().to(device)
        mask_val, mask_test = torch.from_numpy(mask_val).float().to(device), torch.from_numpy(mask_test).float().to(device)

        y_test_pred, energies_test, _ = model(X_test)
        y_test_pred = y_test_pred.cpu().detach().numpy()
        X_test = X_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        mask_test = mask_test.cpu().detach().numpy()

        # Save test results
        np.savez(os.path.join(save_data_dir, Title + '_Test.npz'), X_test=X_test, y_test=y_test, mask_test=mask_test, y_test_pred=y_test_pred)

#%%
# Plot test results
Data = [np.load(os.path.join(save_data_dir, Title + '_Test.npz')) for Title in Titles]

X_tests, y_tests, mask_tests, y_test_preds = zip(*[(dat['X_test'], dat['y_test'], dat['mask_test'], dat['y_test_pred']) for dat in Data])

X_tests = [[X_tests[ii][i, mask_tests[ii][i].astype(int).astype(bool), 0] for i in range(len(X_tests[ii]))] for ii in range(len(X_tests))]
y_tests = [[y_tests[ii][i, mask_tests[ii][i].astype(int).astype(bool)] for i in range(len(y_tests[ii]))] for ii in range(len(y_tests))]
y_test_preds = [[y_test_preds[ii][i, mask_tests[ii][i].astype(int).astype(bool)] for i in range(len(y_test_preds[ii]))] for ii in range(len(y_test_preds))]

fig, axs = plt.subplots(2, 3, figsize=(6, 4))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
for ii in range(len(Titles)):
    axs[ii, 0].plot(X_tests[ii][0], y_tests[ii][0], 'k', linewidth=0.7)
    axs[ii, 0].plot(X_tests[ii][0], y_test_preds[ii][0], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
    axs[ii, 0].grid()
    axs[ii, 1].plot(X_tests[ii][2], y_tests[ii][2], 'k', linewidth=0.7)
    axs[ii, 1].plot(X_tests[ii][2], y_test_preds[ii][2], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
    axs[ii, 1].grid()
    axs[ii, 2].plot(X_tests[ii][10], y_tests[ii][10], 'k', linewidth=0.7)
    axs[ii, 2].plot(X_tests[ii][10], y_test_preds[ii][10], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
    axs[ii, 2].grid()




# %% Response to Cyclic - Bouc-Wen
Title = Titles[0]
mat_type = 'BoucWen'
k0 = 6.283**2
mat_props = [0.1, k0, 1., -0.5, 1.5, 1., 0., 0., 0.]

protocol_seq_len = 20000
periods = [2000]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1, -1, 1/2, 1/4]
slopes = [slope / protocol_seq_len * 1 for slope in slopes]
intercept = 0
disps = []

for period, repetition in zip(periods, repetitions):
    for slope in slopes:
        disp = linear_protocol(slope, intercept, period, repetition)
        disps.append(disp)
        plt.figure()
        plt.plot(disp)
        plt.title(f'period: {period}, slope: {slope}, intercept: {intercept}')
if generate_data:
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_max, y_max, normalize_gap = Data['X_max'], Data['y_max'], Data['normalize_gap']
    # Test for maximum displacement of 1.1 larger than it was in training
    disps = [disp * X_max * (1.1) for disp in disps]

    outputs = [Experiment.static_1DOF(mat_type, mat_props, disps[i]) for i in range(len(disps))]
    X_tests, y_tests = [disp / (X_max * (1 + normalize_gap)) for disp in disps], [output['force'] / (y_max * (1 + normalize_gap)) for output in outputs]
    n_samples = len(disps)

    X_test, y_test, time_length = [], [], []

    num_inputs = 2  # Displacement and mask
    for i in range(n_samples):
        disp, force = X_tests[i], y_tests[i]
        X_test.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y_test.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)

    for i in range(n_samples):
        if len(X_test[i]) < max_time_length:
            X_test[i] = np.concatenate((X_test[i], np.zeros((max_time_length - len(X_test[i]), num_inputs))), axis=0)
            y_test[i] = np.concatenate((y_test[i], np.zeros((max_time_length - len(y_test[i])))), axis=0)
    X_test, y_test = np.array(X_test), np.array(y_test)


    loss_data = pd.read_csv(os.path.normpath(os.path.join(loss_dir, Title + '_loss.csv')))
    losses = loss_data['Loss'].values

    best_model_idx = np.argmin(losses)
    best_model_epoch = loss_data['Epoch'].values[best_model_idx]
    checkpoint = torch.load(os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_{}.pth'.format(int(best_model_epoch)))))

    nn_size = checkpoint['cell.fc_f.bias'].size()[0]
    model = CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load test data
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_test, mask_test = X_test[:, :, :1], X_test[:, :, 1]
    X_test =torch.from_numpy(X_test).float().to(device)
    X_test = add_diff(X_test)
    y_test = torch.from_numpy(y_test).float().to(device)
    mask_test = torch.from_numpy(mask_test).float().to(device)

    y_test_pred, energies_test, _ = model(X_test)
    y_test_pred = y_test_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    mask_test = mask_test.cpu().detach().numpy()

    # Save test results
    np.savez(os.path.join(save_data_dir, Title + '_Cyclic.npz'), X_test=X_test, y_test=y_test, mask_test=mask_test, y_test_pred=y_test_pred)

#%%
# Plot test results
Data = np.load(os.path.join(save_data_dir, Title + '_Cyclic.npz'))

X_test, y_test, mask_test, y_test_pred = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']
X_test = [X_test[i, mask_test[i].astype(int).astype(bool), 0] for i in range(len(X_test))]
y_test = [y_test[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test))]
y_test_pred = [y_test_pred[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test_pred))]





#%% Response to Cyclic - BWBN
Title = Titles[1]
mat_type = 'BWBN'
k0 = 6.283**2
mat_props = [0.1, k0, 1., -0.5, 1.5, 1., 0.1, 0.97, 1., 0.2, 0.002, 0.1, 1.0*10**-4, 10**6]

protocol_seq_len = 20000
periods = [2000]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1, -1, 1/2, 1/4]
slopes = [slope / protocol_seq_len * 1 for slope in slopes]
intercept = 0
disps = []

for period, repetition in zip(periods, repetitions):
    for slope in slopes:
        disp = linear_protocol(slope, intercept, period, repetition)
        disps.append(disp)
        plt.figure()
        plt.plot(disp)
        plt.title(f'period: {period}, slope: {slope}, intercept: {intercept}')
if generate_data:
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_max, y_max, normalize_gap = Data['X_max'], Data['y_max'], Data['normalize_gap']
    # Test for maximum displacement of 1.1 larger than it was in training
    disps = [disp * X_max * (1.1) for disp in disps]

    outputs = [Experiment.static_1DOF(mat_type, mat_props, disps[i]) for i in range(len(disps))]
    X_tests, y_tests = [disp / (X_max * (1 + normalize_gap)) for disp in disps], [output['force'] / (y_max * (1 + normalize_gap)) for output in outputs]
    n_samples = len(disps)

    X_test, y_test, time_length = [], [], []

    num_inputs = 2  # Displacement and mask
    for i in range(n_samples):
        disp, force = X_tests[i], y_tests[i]
        X_test.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y_test.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)

    for i in range(n_samples):
        if len(X_test[i]) < max_time_length:
            X_test[i] = np.concatenate((X_test[i], np.zeros((max_time_length - len(X_test[i]), num_inputs))), axis=0)
            y_test[i] = np.concatenate((y_test[i], np.zeros((max_time_length - len(y_test[i])))), axis=0)
    X_test, y_test = np.array(X_test), np.array(y_test)


    loss_data = pd.read_csv(os.path.normpath(os.path.join(loss_dir, Title + '_loss.csv')))
    losses = loss_data['Loss'].values

    best_model_idx = np.argmin(losses)
    best_model_epoch = loss_data['Epoch'].values[best_model_idx]
    checkpoint = torch.load(os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_{}.pth'.format(int(best_model_epoch)))))

    nn_size = checkpoint['cell.fc_f.bias'].size()[0]
    model = CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load test data
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_test, mask_test = X_test[:, :, :1], X_test[:, :, 1]
    X_test =torch.from_numpy(X_test).float().to(device)
    X_test = add_diff(X_test)
    y_test = torch.from_numpy(y_test).float().to(device)
    mask_test = torch.from_numpy(mask_test).float().to(device)

    y_test_pred, energies_test, _ = model(X_test)
    y_test_pred = y_test_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    mask_test = mask_test.cpu().detach().numpy()

    # Save test results
    np.savez(os.path.join(save_data_dir, Title + '_Cyclic.npz'), X_test=X_test, y_test=y_test, mask_test=mask_test, y_test_pred=y_test_pred)

#%%
# Plot test results
Data = np.load(os.path.join(save_data_dir, Title + '_Cyclic.npz'))

X_test, y_test, mask_test, y_test_pred = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']
X_test = [X_test[i, mask_test[i].astype(int).astype(bool), 0] for i in range(len(X_test))]
y_test = [y_test[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test))]
y_test_pred = [y_test_pred[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test_pred))]

