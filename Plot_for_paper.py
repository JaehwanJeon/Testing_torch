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
EQ_data_file = 'EQ_list.txt'
gm_list, t_list = read_EQ(EQ_data_file)

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
    processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
    save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
    if generate_data:
        test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)

# Plot test results
Data = [np.load(os.path.join(save_data_dir, Title + '_Test.npz')) for Title in Titles]

X_tests, y_tests, mask_tests, y_test_preds = zip(*[(dat['X_test'], dat['y_test'], dat['mask_test'], dat['y_test_pred']) for dat in Data])

X_tests = [[X_tests[ii][i, mask_tests[ii][i].astype(int).astype(bool), 0] for i in range(len(X_tests[ii]))] for ii in range(len(X_tests))]
y_tests = [[y_tests[ii][i, mask_tests[ii][i].astype(int).astype(bool)] for i in range(len(y_tests[ii]))] for ii in range(len(y_tests))]
y_test_preds = [[y_test_preds[ii][i, mask_tests[ii][i].astype(int).astype(bool)] for i in range(len(y_test_preds[ii]))] for ii in range(len(y_test_preds))]

fig, axs = plt.subplots(2, 3, figsize=(6, 4))
plt.subplots_adjust(wspace=0.4, hspace=0.4, right=0.95)
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
axs[0, 0].text(0.5, -0.15, '(a)', transform=axs[0, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 1].text(0.5, -0.15, '(b)', transform=axs[0, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 2].text(0.5, -0.15, '(c)', transform=axs[0, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].text(0.5, -0.3, '(d)', transform=axs[1, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 1].text(0.5, -0.3, '(e)', transform=axs[1, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 2].text(0.5, -0.3, '(f)', transform=axs[1, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].set_xlabel('Normalized displacement')
axs[1, 1].set_xlabel('Normalized displacement')
axs[1, 2].set_xlabel('Normalized displacement')
axs[0, 0].set_ylabel('Normalized force')
axs[1, 0].set_ylabel('Normalized force')
axs[0, 0].legend(['True', 'Predicted'], loc='lower right', fontsize=6.5)
fig.savefig(os.path.join(save_dir, 'BoucWen_PE_PI_DA_BWBN_PE_PI_DA_Test.svg'), bbox_inches='tight')


# %% 
# Response to Cyclic - Bouc-Wen
Title = Titles[0]
mat_type = 'BoucWen'
k0 = 6.283**2
mat_props = [0.1, k0, 1., -0.5, 1.5, 1., 0., 0., 0.]

protocol_seq_len = 3000
periods = [300]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1/4, 1/2, 1, -1]
slopes = [slope / protocol_seq_len * 1 for slope in slopes]
intercept = 0
disps = []

for period, repetition in zip(periods, repetitions):
    for slope in slopes:
        disp = linear_protocol(slope, intercept, period, repetition)
        disps.append(disp)

processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
processed_cyclic_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_cyclic.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Cyclic.npz')

if generate_data:
    generate_cyclic(mat_type,
                    mat_props,
                    disps,
                    processed_data_path,
                    processed_cyclic_path)

    test_prediction(loss_dir, Title, model_dir, processed_cyclic_path, save_data_path, device)

# Response to Cyclic - BWBN
Title = Titles[1]
mat_type = 'BWBN'
k0 = 6.283**2
mat_props = [0.1, k0, 1., -0.5, 1.5, 1., 0.1, 0.97, 1., 0.2, 0.002, 0.1, 1.0*10**-4, 10**6]

protocol_seq_len = 20000
periods = [2000]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1/4, 1/2, 1, -1]
slopes = [slope / protocol_seq_len * 1 for slope in slopes]
intercept = 0
disps = []

for period, repetition in zip(periods, repetitions):
    for slope in slopes:
        disp = linear_protocol(slope, intercept, period, repetition)
        disps.append(disp)

fig, ax = plt.subplots(1, 1, figsize=(6, 2))
ax.plot(disps[0], linestyle='-', linewidth=1)
ax.plot(disps[1], linestyle='--', linewidth=1)
ax.plot(disps[2], linestyle='-.', linewidth=1)
ax.plot(disps[3], linestyle=':', linewidth=1)
ax.grid()
ax.set_xlabel('Step')
ax.set_ylabel('Normalized displacement')
ax.set_xlim([0, protocol_seq_len])
ax.legend(['1/4', '1/2', '1', '-1'], loc='upper left', fontsize=7)
fig.savefig(os.path.join(save_dir, 'Cyclic_protocol.svg'), bbox_inches='tight')

processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
processed_cyclic_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_cyclic.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Cyclic.npz')

if generate_data:
    generate_cyclic(mat_type,
                    mat_props,
                    disps,
                    processed_data_path,
                    processed_cyclic_path)

    test_prediction(loss_dir, Title, model_dir, processed_cyclic_path, save_data_path, device)

# Plot test results
X_tests, y_tests, y_test_preds = [], [], []
for Title in Titles:
    save_data_path = os.path.join(save_data_dir, Title + '_Cyclic.npz')
    Data = np.load(save_data_path)
    X_test, y_test, mask_test, y_test_pred = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']
    X_test = [X_test[i, mask_test[i].astype(int).astype(bool), 0] for i in range(len(X_test))]
    y_test = [y_test[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test))]
    y_test_pred = [y_test_pred[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test_pred))]              
    X_tests.append(X_test), y_tests.append(y_test), y_test_preds.append(y_test_pred)

fig, axs = plt.subplots(2, 4, figsize=(6, 2.9))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for ii in range(len(Titles)):
    for jj in range(len(disps)):
        axs[ii, jj].plot(X_tests[ii][jj], y_tests[ii][jj], 'k', linewidth=0.7)
        axs[ii, jj].plot(X_tests[ii][jj], y_test_preds[ii][jj], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
        axs[ii, jj].grid()
axs[0, 0].text(0.5, -0.22, '(a)', transform=axs[0, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 0].legend(['True', 'Predicted'], loc='lower right', fontsize=4.55)
axs[0, 1].text(0.5, -0.22, '(b)', transform=axs[0, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 2].text(0.5, -0.22, '(c)', transform=axs[0, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 3].text(0.5, -0.22, '(d)', transform=axs[0, 3].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].text(0.5, -0.22, '(e)', transform=axs[1, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 1].text(0.5, -0.22, '(f)', transform=axs[1, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 2].text(0.5, -0.22, '(g)', transform=axs[1, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 3].text(0.5, -0.22, '(h)', transform=axs[1, 3].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
fig.text(0.5, -0.05, 'Normalized displacement', ha='center')
fig.text(0.05, 0.5, 'Normalized force', va='center', rotation='vertical')
fig.savefig(os.path.join(save_dir, 'BoucWen_PE_PI_DA_Cyclic_BWBN_PE_PI_DA_Cyclic.svg'), bbox_inches='tight')





# %% Plot non-Bouc-Wen class test results
Titles = ['Bilinear_PE_PI_DA', 'RO_PE_PI_DA']
# Load best model
for Title in Titles:
    processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
    save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
    if generate_data:
        test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)

# Plot test results
Data = [np.load(os.path.join(save_data_dir, Title + '_Test.npz')) for Title in Titles]

X_tests, y_tests, mask_tests, y_test_preds = zip(*[(dat['X_test'], dat['y_test'], dat['mask_test'], dat['y_test_pred']) for dat in Data])

X_tests = [[X_tests[ii][i, mask_tests[ii][i].astype(int).astype(bool), 0] for i in range(len(X_tests[ii]))] for ii in range(len(X_tests))]
y_tests = [[y_tests[ii][i, mask_tests[ii][i].astype(int).astype(bool)] for i in range(len(y_tests[ii]))] for ii in range(len(y_tests))]
y_test_preds = [[y_test_preds[ii][i, mask_tests[ii][i].astype(int).astype(bool)] for i in range(len(y_test_preds[ii]))] for ii in range(len(y_test_preds))]

fig, axs = plt.subplots(2, 3, figsize=(6, 4))
plt.subplots_adjust(wspace=0.4, hspace=0.4, right=0.95)
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
axs[0, 0].text(0.5, -0.15, '(a)', transform=axs[0, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 1].text(0.5, -0.15, '(b)', transform=axs[0, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 2].text(0.5, -0.15, '(c)', transform=axs[0, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].text(0.5, -0.3, '(d)', transform=axs[1, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 1].text(0.5, -0.3, '(e)', transform=axs[1, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 2].text(0.5, -0.3, '(f)', transform=axs[1, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].set_xlabel('Normalized displacement')
axs[1, 1].set_xlabel('Normalized displacement')
axs[1, 2].set_xlabel('Normalized displacement')
axs[0, 0].set_ylabel('Normalized force')
axs[1, 0].set_ylabel('Normalized force')
axs[0, 0].legend(['True', 'Predicted'], loc='lower right', fontsize=6.5)
fig.savefig(os.path.join(save_dir, 'Bilinear_PE_PI_DA_RO_PE_PI_DA_Test.svg'), bbox_inches='tight')


# %% 
# Response to Cyclic - Bilinear
Title = Titles[0]
mat_type = 'Steel01'
mat_props = [7., 6.283**2, 0.7]

protocol_seq_len = 3000
periods = [300]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1/4, 1/2, 1, -1]
slopes = [slope / protocol_seq_len * 1 for slope in slopes]
intercept = 0
disps = []

for period, repetition in zip(periods, repetitions):
    for slope in slopes:
        disp = linear_protocol(slope, intercept, period, repetition)
        disps.append(disp)

processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
processed_cyclic_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_cyclic.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Cyclic.npz')

if generate_data:
    generate_cyclic(mat_type,
                    mat_props,
                    disps,
                    processed_data_path,
                    processed_cyclic_path)

    test_prediction(loss_dir, Title, model_dir, processed_cyclic_path, save_data_path, device)

# Response to Cyclic - Ramberg-Osgood
Title = Titles[1]
mat_type = 'RambergOsgoodSteel'
k0 = 6.283**2
mat_props = [7.5, 6.283**2, 0.002, 5]

protocol_seq_len = 3500
periods = [350]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1/4, 1/2, 1, -1]
slopes = [slope / protocol_seq_len * 1 for slope in slopes]
intercept = 0
disps = []

for period, repetition in zip(periods, repetitions):
    for slope in slopes:
        disp = linear_protocol(slope, intercept, period, repetition)
        disps.append(disp)

processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
processed_cyclic_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_cyclic.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Cyclic.npz')

if generate_data:
    generate_cyclic(mat_type,
                    mat_props,
                    disps,
                    processed_data_path,
                    processed_cyclic_path)

    test_prediction(loss_dir, Title, model_dir, processed_cyclic_path, save_data_path, device)

# Plot test results
X_tests, y_tests, y_test_preds = [], [], []
for Title in Titles:
    save_data_path = os.path.join(save_data_dir, Title + '_Cyclic.npz')
    Data = np.load(save_data_path)
    X_test, y_test, mask_test, y_test_pred = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']
    X_test = [X_test[i, mask_test[i].astype(int).astype(bool), 0] for i in range(len(X_test))]
    y_test = [y_test[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test))]
    y_test_pred = [y_test_pred[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test_pred))]              
    X_tests.append(X_test), y_tests.append(y_test), y_test_preds.append(y_test_pred)

fig, axs = plt.subplots(2, 4, figsize=(6, 2.9))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for ii in range(len(Titles)):
    for jj in range(len(disps)):
        axs[ii, jj].plot(X_tests[ii][jj], y_tests[ii][jj], 'k', linewidth=0.7)
        axs[ii, jj].plot(X_tests[ii][jj], y_test_preds[ii][jj], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
        axs[ii, jj].grid()
axs[0, 0].text(0.5, -0.22, '(a)', transform=axs[0, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 0].legend(['True', 'Predicted'], loc='lower right', fontsize=4.55)
axs[0, 1].text(0.5, -0.22, '(b)', transform=axs[0, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 2].text(0.5, -0.22, '(c)', transform=axs[0, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 3].text(0.5, -0.22, '(d)', transform=axs[0, 3].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].text(0.5, -0.22, '(e)', transform=axs[1, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 1].text(0.5, -0.22, '(f)', transform=axs[1, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 2].text(0.5, -0.22, '(g)', transform=axs[1, 2].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 3].text(0.5, -0.22, '(h)', transform=axs[1, 3].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
fig.text(0.5, -0.05, 'Normalized displacement', ha='center')
fig.text(0.05, 0.5, 'Normalized force', va='center', rotation='vertical')
fig.savefig(os.path.join(save_dir, 'Bilinear_PE_PI_DA_Cyclic_RO_PE_PI_DA_Cyclic.svg'), bbox_inches='tight')





# %%
Title = 'IMK_PE_PI_DA'
processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
Data = np.load(processed_data_path)
X_test, y_test = Data['X_test'], Data['y_test']
del Data
X_test, mask_test = X_test[:, :, :1], X_test[:, :, 1]
X_test = [X_test[i, mask_test[i].astype(int).astype(bool), 0] for i in range(len(X_test))]
y_test = [y_test[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test))]
# %%
i = 27
plt.plot(X_test[i], y_test[i], '*')
plt.plot(X_test[i], y_test[i], 'k', linewidth=0.7, alpha=0.7)