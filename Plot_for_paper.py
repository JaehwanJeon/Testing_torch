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
from sklearn.metrics import r2_score
import scipy.io


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




# %% LIRAND
# Load data
disps = []
forces = []
BW_forces = []
for i in range(3):
    Data = np.load(os.path.join(save_data_dir, 'Link_{}.npz'.format(i)))
    disps.append(Data['train_disps'])
    forces.append(Data['train_forces'])
    BW_forces.append(Data['BW_force'])

fig, axs = plt.subplots(1, 3, figsize=(6, 1.8))
plt.subplots_adjust(wspace=0.32)
for i in range(3):
    axs[i].plot(disps[i]*711.2, forces[i], 'k', linewidth=0.7, label='Reference')
    axs[i].plot(disps[i]*711.2, BW_forces[i], 'orange', linewidth=0.7, linestyle='--', alpha=0.8, label='BW')
    axs[i].grid()
    # axs[i].set_xlim([-0.12, 0.12])
    # axs[i].set_ylim([-1.2, 1.2])
    # axs[i].text(0.5, -0.25, '({})'.format(chr(97+i)), transform=axs[i].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
    axs[i].set_xlabel('Displacement [mm]')
axs[0].set_ylim([-750, 982])
axs[0].set_ylabel('Force [kN]')
axs[0].legend(loc='upper left', fontsize=5.5, ncol=2)
axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
fig.savefig(os.path.join(save_dir, 'LIRAND.svg'), bbox_inches='tight')



# %% LICYC
# Load data
Data = np.load(os.path.join(save_data_dir, 'Link.npz'))
disp = Data['val_disps']
force = Data['val_forces']
BW_force = Data['BW_force']
pretrained_force = Data['pretrained_force']
trained_force = Data['trained_force']

fig, axs = plt.subplots(1, 2, figsize=(4, 1.8))
plt.subplots_adjust(wspace=0.32)

# 첫 번째 서브플롯 (axs[0])
line1, = axs[0].plot(disp*711.2, force, 'k', linewidth=0.7, label='Reference')
line2, = axs[0].plot(disp*711.2, BW_force, 'orange', linewidth=0.7, linestyle='--', alpha=0.7, label='BW')
line3, = axs[0].plot(disp*711.2, pretrained_force, 'blue', linewidth=0.7, linestyle='--', alpha=0.7, label='Pretrained')
axs[0].grid()
axs[0].set_xlabel('Displacement [mm]')
axs[0].set_ylabel('Force [kN]')

# 두 번째 서브플롯 (axs[1])
line4, = axs[1].plot(disp*711.2, force, 'k', linewidth=0.7)
line5, = axs[1].plot(disp*711.2, BW_force, 'orange', linewidth=0.7, linestyle='--', alpha=0.7)
line6, = axs[1].plot(disp*711.2, trained_force, 'r', linewidth=0.7, linestyle='--', alpha=0.7, label='Trained')

axs[1].grid()
axs[1].set_xlabel('Displacement [mm]')
axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# Figure 전체에 대한 legend 추가
fig.legend([line1, line2, line3, line6], ['Reference', 'BW', 'Pretrained', 'Trained'], 
           loc='upper center', ncol=4, fontsize=5.5, bbox_to_anchor=(0.565, 1.0))

fig.savefig(os.path.join(save_dir, 'LICYC.svg'), bbox_inches='tight')

MSE_BW = np.sum((force - BW_force) ** 2) / len(force)
MSE_trained = np.sum((force - trained_force) ** 2) / len(force)
print('MSE_BW: {:.4e}'.format(MSE_BW))
print('MSE_trained: {:.4e}'.format(MSE_trained))
print('Ratio: {:.4f}'.format(MSE_trained / MSE_BW))



# %% OPFD
Data = np.load('Result Plots/OP_4_PE_VelNorm_data.npz')
Data.keys()
X_test = Data['X_val']
y_test = Data['y_val']
y_test_pred = Data['y_val_pred']
mask_test = Data['mask_val']
idc = [4, 6, 9]
fig, axs = plt.subplots(1, 3, figsize=(6, 1.8))
plt.subplots_adjust(wspace=0.32)
for i in range(len(idc)):
    axs[i].plot(X_test[idc[i], :, 0], y_test[idc[i], :], 'k', linewidth=0.7)
    axs[i].plot(X_test[idc[i], :, 0], y_test_pred[idc[i], :], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
    axs[i].grid()
axs[1].legend(['Reference', 'Predicted'], loc='lower right', fontsize=5.5)
axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[1].set_xlabel('Normalized displacement')
axs[0].set_ylabel('Normalized force')
fig.savefig(os.path.join(save_dir, 'OPFD.svg'), bbox_inches='tight')
# X_train, y_train = Data['X_train'], Data['y_train']
# X_valid, y_valid = Data['X_valid'], Data['y_valid']
# X_test, y_test = Data['X_test'], Data['y_test']
# i = 8
# fig, axs = plt.subplots(2, 2, figsize=(6, 4))
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# axs[0, 0].plot(X_train[4*i, :, 0], y_train[4*i, :, 0], 'k', linewidth=0.7, marker='o', markersize=2)
# axs[0, 0].set_title('{}'.format(4*i))
# axs[0, 1].plot(X_train[4*i+1, :, 0], y_train[4*i+1, :, 0], 'k', linewidth=0.7, marker='o', markersize=2)
# axs[0, 1].set_title('{}'.format(4*i+1))
# axs[1, 0].plot(X_train[4*i+2, :, 0], y_train[4*i+2, :, 0], 'k', linewidth=0.7, marker='o', markersize=2)
# axs[1, 0].set_title('{}'.format(4*i+2))
# axs[1, 1].plot(X_train[4*i+3, :, 0], y_train[4*i+3, :, 0], 'k', linewidth=0.7, marker='o', markersize=2)
# axs[1, 1].set_title('{}'.format(4*i+3))





# %% BW
if generate_data:
    Titles = ['BoucWen_PE_PI_DA_VelNorm', 'BWBN_PE_PI_DA_VelNorm']
    ids = [2, 10]
    Datas = [np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))) for Title in Titles]
    X_tests = [Data['X_test'] for Data in Datas]
    y_tests = [Data['y_test'] for Data in Datas]
    mask_tests = [y_test[:, :, 1:2] for y_test in y_tests]
    y_tests = [y_test[:, :, 0:1] for y_test in y_tests]
    Xs = [X_test[ids[i], mask_tests[i][ids[i], :, 0].astype(int).astype(bool), 0] for i, X_test in enumerate(X_tests)]
    ys = [y_test[ids[i], mask_tests[i][ids[i], :, 0].astype(int).astype(bool), 0] for i, y_test in enumerate(y_tests)]
    np.savez(os.path.join(save_data_dir, 'BWSamples.npz'), BW_X = Xs[0], BW_y = ys[0], BWBN_X = Xs[1], BWBN_y = ys[1])

Data = np.load(os.path.join(save_data_dir, 'BWSamples.npz'))
BW_X, BW_y, BWBN_X, BWBN_y = Data['BW_X'], Data['BW_y'], Data['BWBN_X'], Data['BWBN_y']
BW_X, BW_y = BW_X[:20000], BW_y[:20000]
fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.2))
axs.plot(BW_X, BW_y, 'k', linewidth=0.4)
axs.set_xlabel('Displacement')
axs.set_ylabel('Force')
axs.set_xticks([])
axs.set_yticks([])
# axs[0].grid()
fig.savefig(os.path.join(save_dir, 'BW_a.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.2))
axs.plot(BWBN_X, BWBN_y, 'k', linewidth=0.4)
axs.set_xlabel('Displacement')
axs.set_ylabel('Force')
axs.set_xticks([])
axs.set_yticks([])
# axs[1].grid()
# axs[0].text(0.5, -0.2, '(a)', transform=axs[0].transAxes, fontsize=8, fontweight='bold', va='top', ha='center')
# axs[1].text(0.5, -0.2, '(b)', transform=axs[1].transAxes, fontsize=8, fontweight='bold', va='top', ha='center')
fig.savefig(os.path.join(save_dir, 'BW_b.svg'), bbox_inches='tight')




# %% Plot @RATE
Data = np.load(os.path.join(save_data_dir, 'BWSamples.npz'))
X, y = Data['BW_X'], Data['BW_y']
X, y = X[:20000], y[:20000]
sample_rate_0, sample_rate_1 = 11, 21
fig, axs = plt.subplots(2, 2, figsize=(6, 3))
plt.subplots_adjust(wspace=0.25, hspace=0.5)
axs[0, 0].plot(X[::sample_rate_0], 'k', linewidth=0.7)
axs[0, 0].set_xlim([0, len(X[::sample_rate_0])])
axs[0, 0].set_ylabel('Displacement')
axs[0, 0].set_yticklabels([])
axs[0, 0].text(0.5, -0.25, '(a)', transform=axs[0, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[0, 1].plot(X[::sample_rate_1], 'k', linewidth=0.7)
axs[0, 1].set_xlim([0, len(X[::sample_rate_0])])
axs[0, 1].set_yticklabels([])
axs[0, 1].text(0.5, -0.25, '(b)', transform=axs[0, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].plot(y[::sample_rate_0], 'k', linewidth=0.7)
axs[1, 0].set_xlim([0, len(y[::sample_rate_0])])
axs[1, 0].set_ylabel('Force')
axs[1, 0].set_yticklabels([])
axs[1, 0].text(0.5, -0.4, '(c)', transform=axs[1, 0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1, 0].set_xlabel('Step')
axs[1, 1].plot(y[::sample_rate_1], 'k', linewidth=0.7)
axs[1, 1].set_xlim([0, len(y[::sample_rate_0])])
axs[1, 1].set_xlabel('Step')
axs[1, 1].set_yticklabels([])
axs[1, 1].text(0.5, -0.4, '(d)', transform=axs[1, 1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')

fig.savefig(os.path.join(save_dir, 'RATE_a.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 2, figsize=(6, 2.8))
axs[0].plot(X, y, 'k', linewidth=0.7, alpha=0.5)
axs[0].plot(X[::sample_rate_0], y[::sample_rate_0], '*', alpha=0.5, markersize=2, color='r')
axs[0].set_xlabel('Displacement')
axs[0].set_ylabel('Force')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].text(0.5, -0.15, '(a)', transform=axs[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
axs[1].plot(X, y, 'k', linewidth=0.7, alpha=0.5)
axs[1].plot(X[::sample_rate_1], y[::sample_rate_1], '*', alpha=0.5, markersize=2, color='b')
axs[1].set_xlabel('Displacement')
axs[1].set_ylabel('Force')
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].text(0.5, -0.15, '(b)', transform=axs[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='center')

fig.savefig(os.path.join(save_dir, 'RATE_b.svg'), bbox_inches='tight')




# %% Plot @ALG
from matplotlib.collections import LineCollection
if generate_data:
    Title = 'BWBN_PE_PI_DA_VelNorm'
    id = 9
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_test = Data['X_test']
    y_test = Data['y_test']
    mask_test = y_test[:, :, 1:2]
    y_test = y_test[:, :, 0:1]
    Xs = X_test[id, mask_test[id, :, 0].astype(int).astype(bool), 0]
    ys = y_test[id, mask_test[id, :, 0].astype(int).astype(bool), 0]
    Xs = Xs[:10000:10]
    ys = ys[:10000:10]
    np.savez(os.path.join(save_data_dir, 'ALGSamples.npz'), X = Xs, y = ys)

Data = np.load(os.path.join(save_data_dir, 'ALGSamples.npz'))
X, y = Data['X'], Data['y']
fig, axs = plt.subplots(1, 1, figsize=(5, 1))
threshold = 0.05
# 선을 세그먼트로 나누기 위한 준비
points = np.array([np.arange(len(y)), y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 선의 색을 기준선 위와 아래로 나누기
colors = ['k' if y_value >= threshold else 'k' for y_value in y[:-1]]

# LineCollection을 사용해 세그먼트를 그리기
lc = LineCollection(segments, colors=colors, linewidth=0.7)
axs.add_collection(lc)
axs.set_xticks([])
axs.set_yticks([])
axs.set_xlim([0, 1000])
axs.set_ylim([-0.4, 0.3])
axs.axhline(threshold, color='k', linewidth=0.7, linestyle='--')
axs.hlines([0.25, 0.2, -0.15, -0.2, -0.25], 0, 1000, color='grey', linewidth=0.5, linestyle='--')

fig.savefig(os.path.join(save_dir, 'ALGsamplek.svg'), bbox_inches='tight')

time_length = len(y)
chop_vector = np.array([0.2, 0.1, -0.1, -0.2])

# Expand the mask and tensors along the second axis
fig, ax = plt.subplots(4, 1, figsize=(5, 3))
for i, chop_value in enumerate(chop_vector):
    sign_chop = ((y - chop_value)>0).astype(int)
    ax[i].plot(sign_chop, linewidth=0.7, color='k')
for i in range(4):
    ax[i].set_xticklabels([])
    ax[i].set_yticks([0, 1])
# ax[3].set_yticks([0, 1])
for i in range(4):
    ax[i].set_yticklabels(['–', '+'])
    ax[i].set_xlim([0, 1000])
    ax[i].set_ylim([-0.1, 1.1])
ax[3].set_xlabel('Step')

fig.savefig(os.path.join(save_dir, 'ALG.svg'), bbox_inches='tight')




# %% Plot @AUG
if generate_data:
    Title = 'BWBN_PE_PI_DA_VelNorm'
    id = 9
    Data = np.load(os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')))
    X_test = Data['X_test']
    y_test = Data['y_test']
    mask_test = y_test[:, :, 1:2]
    y_test = y_test[:, :, 0:1]
    Xs = X_test[id, mask_test[id, :, 0].astype(int).astype(bool), 0]
    ys = y_test[id, mask_test[id, :, 0].astype(int).astype(bool), 0]
    Xs = Xs[500:4500:4]
    ys = ys[500:4500:4]
    np.savez(os.path.join(save_data_dir, 'SAMPSamples.npz'), X = Xs, y = ys)


Data = np.load(os.path.join(save_data_dir, 'SAMPSamples.npz'))
X = Data['X']
y = Data['y']

fig, axs = plt.subplots(1, 1, figsize=(5, 1.5/2))
plt.subplots_adjust(wspace=0.5)
axs.plot(X, 'orange', linewidth=0.7, label='Displacement')
axs.plot(y, 'blue', linewidth=0.7, linestyle='--', label='Force')
axs.set_xlim([0, 1000])
axs.set_xticks([])
axs.set_yticks([])
axs.legend(loc='lower left', fontsize=5.5, ncol=2, bbox_to_anchor=(0.63, 1.0))
axs.set_xlabel('Time step')
axs.set_ylabel('Displacement / Force', fontsize=6)

fig.savefig(os.path.join(save_dir, 'AUG_a.svg'), bbox_inches='tight')


augmentation_rate = 0.7
fig, axs = plt.subplots(3, 1, figsize=(4, 1.5*3/2))
plt.subplots_adjust(wspace=0.5)
augmented_idx = np.arange(len(X))
augmented_idx = np.random.choice(augmented_idx, int(len(X) * augmentation_rate), replace=False)
augmented_idx = np.sort(augmented_idx)
Xp = X[augmented_idx]
yp = y[augmented_idx]

axs[0].set_xlim([0, int(1000 * augmentation_rate)])
axs[0].plot(Xp, 'orange', linewidth=0.7)
axs[0].plot(yp, 'blue', linewidth=0.7, linestyle='--')
axs[0].set_xticks([])
axs[0].set_yticks([])

augmented_idx = np.arange(len(X))
augmented_idx = np.random.choice(augmented_idx, int(len(X) * augmentation_rate), replace=False)
augmented_idx = np.sort(augmented_idx)
Xp = X[augmented_idx]
yp = y[augmented_idx]

axs[1].plot(Xp, 'orange', linewidth=0.7)
axs[1].plot(yp, 'blue', linewidth=0.7, linestyle='--')
axs[1].set_xlim([0, int(1000 * augmentation_rate)])
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_ylabel('Displacement / Force')

augmented_idx = np.arange(len(X))
augmented_idx = np.random.choice(augmented_idx, int(len(X) * augmentation_rate), replace=False)
augmented_idx = np.sort(augmented_idx)
Xp = X[augmented_idx]
yp = y[augmented_idx]

axs[2].plot(Xp, 'orange', linewidth=0.7)
axs[2].plot(yp, 'blue', linewidth=0.7, linestyle='--')
axs[2].set_xlim([0, int(1000 * augmentation_rate)])
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].set_xlabel(r'$f_{aug}$ $\times$ Time step')

fig.savefig(os.path.join(save_dir, 'AUG_b.svg'), bbox_inches='tight')


# %% Plot @SAMP

Data = np.load(os.path.join(save_data_dir, 'ALGSamples.npz'))
X = Data['X']
dX = np.diff(X)
augmented_idx = np.arange(len(X))
augmented_idx = np.random.choice(augmented_idx, int(len(X) * 0.7), replace=False)
augmented_idx = np.sort(augmented_idx)
Xp = X[augmented_idx]
dXp = np.diff(Xp)

fig, axs = plt.subplots(1, 1, figsize=(2, 1.5))
plt.subplots_adjust(wspace=0.5)
bins = np.linspace(-0.02, 0.02, 20)
axs.hist(dX, linewidth=0.7, alpha=0.5, bins=bins)
axs.set_xlim([-0.02, 0.02])
axs.set_ylabel('Count')
axs.set_ylim([0, 400])
axs.set_xlabel(r'Normalized $\Delta u$')
fig.savefig(os.path.join(save_dir, 'SAMP_a.svg'), bbox_inches='tight')

# axs.text(0.5, -0.3, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')

fig, axs = plt.subplots(1, 1, figsize=(2, 1.5))
axs.hist(dXp, linewidth=0.7, alpha=0.5, bins=bins)
axs.set_xlim([-0.02, 0.02])
axs.set_ylim([0, 400])
axs.set_ylabel('Count')
axs.set_xlabel(r'Normalized $\Delta u$')
# axs[1].text(0.5, -0.3, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# fig.text(0.5, -0.1, 'Normalized displacement difference', ha='center')
fig.savefig(os.path.join(save_dir, 'SAMP_b.svg'), bbox_inches='tight')





#%% Plot EQ
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
fig.savefig(os.path.join(save_dir, 'GM.svg'), bbox_inches='tight')





# %% Plot BWFD
Title = 'BoucWen_PE_PI_DA'


processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
if generate_data:
    test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)

# Plot test results
Data = np.load(os.path.join(save_data_dir, Title + '_Test.npz'))

X_tests, y_tests, mask_tests, y_test_preds = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']

X_tests = [X_tests[i, mask_tests[i].astype(int).astype(bool), 0] for i in range(len(X_tests))]
y_tests = [y_tests[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_tests))]
y_test_preds = [y_test_preds[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_test_preds))]

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
# plt.subplots_adjust(wspace=0.32)
idc = [1, 8, 10]
axs.plot(X_tests[idc[0]], y_tests[idc[0]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[0]], y_test_preds[idc[0]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_ylabel('Normalized $f$')
axs.set_xlabel('Normalized $u$')
axs.legend(['Reference', 'Predicted'], loc='lower right', fontsize=6.5)
fig.savefig(os.path.join(save_dir, 'BWFD_a.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.plot(X_tests[idc[1]], y_tests[idc[1]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[1]], y_test_preds[idc[1]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
fig.savefig(os.path.join(save_dir, 'BWFD_b.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.plot(X_tests[idc[2]], y_tests[idc[2]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[2]], y_test_preds[idc[2]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
fig.savefig(os.path.join(save_dir, 'BWFD_c.svg'), bbox_inches='tight')
# axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')


# axs[0].legend(['Reference', 'Predicted'], loc='lower right', fontsize=6.5)
# fig.savefig(os.path.join(save_dir, 'BWFD.svg'), bbox_inches='tight')


# %% BW Loss
Title = 'BoucWen_PE_PI_DA'
if generate_data:
    Data = np.load(os.path.join(save_data_dir, Title + '_Test.npz'))
    X_tests, y_tests, mask_tests, y_test_preds, energies_test = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred'][:, :, 0], Data['energies_test'][:, :, 0]
    mse_loss = np.sum((y_tests - y_test_preds) ** 2 * mask_tests) / np.sum(mask_tests)
    phys_loss = drucker_loss_numpy(y_test_preds, energies_test, mask_tests)
    np.savez(os.path.join(save_data_dir, Title + '_Loss.npz'), mse_loss=mse_loss, phys_loss=phys_loss)

Data = np.load(os.path.join(save_data_dir, Title + '_Loss.npz'))
print('MSE loss: {:.4e}'.format(Data['mse_loss']))
print('Physics loss: {:.4e}'.format(Data['phys_loss']))
# %% BWCY
# Response to Cyclic - Bouc-Wen
Title = 'BoucWen_PE_PI_DA'
mat_type = 'BoucWen'
k0 = 6.283**2
mat_props = [0.1, k0, 1., -0.5, 1.5, 1., 0., 0., 0.]

protocol_seq_len = 3000
periods = [300]
repetitions = [int(protocol_seq_len/period) for period in periods]
slopes = [1/4, 1/2, 3/4, 1]
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
ax.legend(['1/4', '1/2', '3/4', '1'], loc='upper left', fontsize=7)
fig.savefig(os.path.join(save_dir, 'CY.svg'), bbox_inches='tight')

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

save_data_path = os.path.join(save_data_dir, Title + '_Cyclic.npz')
Data = np.load(save_data_path)
X_test, y_test, mask_test, y_test_pred = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']
X_test = [X_test[i, mask_test[i].astype(int).astype(bool), 0] for i in range(len(X_test))]
y_test = [y_test[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test))]
y_test_pred = [y_test_pred[i, mask_test[i].astype(int).astype(bool)] for i in range(len(y_test_pred))]              

fig, axs = plt.subplots(1, 1, figsize=(1.3, 1.3))
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
axs.plot(X_test[0], y_test[0], 'k', linewidth=0.7)
axs.plot(X_test[0], y_test_pred[0], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
axs.legend(['Reference', 'Model'], loc='lower right', fontsize=4.59)
fig.savefig(os.path.join(save_dir, 'BWCY_a.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.3, 1.3))
axs.plot(X_test[1], y_test[1], 'k', linewidth=0.7)
axs.plot(X_test[1], y_test_pred[1], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
axs.grid()
fig.savefig(os.path.join(save_dir, 'BWCY_b.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.3, 1.3))
axs.plot(X_test[2], y_test[2], 'k', linewidth=0.7)
axs.plot(X_test[2], y_test_pred[2], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
fig.savefig(os.path.join(save_dir, 'BWCY_c.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.3, 1.3))
axs.plot(X_test[3], y_test[3], 'k', linewidth=0.7)
axs.plot(X_test[3], y_test_pred[3], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
fig.savefig(os.path.join(save_dir, 'BWCY_d.svg'), bbox_inches='tight')


# axs[1].text(0.5, -0.22, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2].text(0.5, -0.22, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[3].text(0.5, -0.22, '(d)', transform=axs[3].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# fig.text(0.5, -0.28, 'Normalized displacement', ha='center')
# fig.text(0.05, 0.5, 'Normalized force', va='center', rotation='vertical')
# fig.savefig(os.path.join(save_dir, 'BWCY.svg'), bbox_inches='tight')


# %% BWBNFD
Titles = ['BWBN_PE_PI_DA', 'BWBN_PyLSTM', 'BWBN_LSTM']

processed_data_paths = [os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz')) for Title in Titles]
save_data_paths = [os.path.join(save_data_dir, Title + '_Test.npz') for Title in Titles]

if generate_data:
    test_prediction(loss_dir, Titles[0], model_dir, processed_data_paths[0], save_data_paths[0], device)
    model = PyLSTM(2, 64, 1)
    test_prediction(loss_dir, Titles[1], model_dir, processed_data_paths[1], save_data_paths[1], device, model)
    model = BasicLSTM(2, 64, 1)
    test_prediction(loss_dir, Titles[2], model_dir, processed_data_paths[2], save_data_paths[2], device, model)

# Plot test results
Data_CustomLSTM = np.load(os.path.join(save_data_dir, Titles[0] + '_Test.npz'))
Data_PyLSTM = np.load(os.path.join(save_data_dir, Titles[1] + '_Test.npz'))
Data_LSTM = np.load(os.path.join(save_data_dir, Titles[2] + '_Test.npz'))

# Plot test results

X_tests_C, y_tests_C, mask_tests_C, y_test_preds_C = Data_CustomLSTM['X_test'], Data_CustomLSTM['y_test'], Data_CustomLSTM['mask_test'], Data_CustomLSTM['y_test_pred']
X_tests_P, y_tests_P, mask_tests_P, y_test_preds_P = Data_PyLSTM['X_test'], Data_PyLSTM['y_test'], Data_PyLSTM['mask_test'], Data_PyLSTM['y_test_pred']
X_tests_L, y_tests_L, mask_tests_L, y_test_preds_L = Data_LSTM['X_test'], Data_LSTM['y_test'], Data_LSTM['mask_test'], Data_LSTM['y_test_pred']

X_tests_C = [X_tests_C[i, mask_tests_C[i].astype(int).astype(bool), 0] for i in range(len(X_tests_C))]
y_tests_C = [y_tests_C[i, mask_tests_C[i].astype(int).astype(bool)] for i in range(len(y_tests_C))]
y_test_preds_C = [y_test_preds_C[i, mask_tests_C[i].astype(int).astype(bool)] for i in range(len(y_test_preds_C))]

X_tests_P = [X_tests_P[i, mask_tests_P[i].astype(int).astype(bool), 0] for i in range(len(X_tests_P))]
y_tests_P = [y_tests_P[i, mask_tests_P[i].astype(int).astype(bool)] for i in range(len(y_tests_P))]
y_test_preds_P = [y_test_preds_P[i, mask_tests_P[i].astype(int).astype(bool)] for i in range(len(y_test_preds_P))]

X_tests_L = [X_tests_L[i, mask_tests_L[i].astype(int).astype(bool), 0] for i in range(len(X_tests_L))]
y_tests_L = [y_tests_L[i, mask_tests_L[i].astype(int).astype(bool)] for i in range(len(y_tests_L))]
y_test_preds_L = [y_test_preds_L[i, mask_tests_L[i].astype(int).astype(bool)] for i in range(len(y_test_preds_L))]

fig1, axs1 = plt.subplots(1, 1, figsize=(1.8, 1.8))
fig2, axs2 = plt.subplots(1, 1, figsize=(1.8, 1.8))
fig3, axs3 = plt.subplots(1, 1, figsize=(1.8, 1.8))

idc = [1, 8, 10]
i = idc[0]
axs1.plot(X_tests_C[i], y_tests_C[i], 'k', linewidth=0.7, label='Reference')
axs1.plot(X_tests_C[i], y_test_preds_C[i], 'r', linewidth=0.7, linestyle='--', alpha=0.7, label='Proposed')
axs1.set_xlabel('Normalized $u$')
axs1.set_ylabel('Normalized $f$')
axs1.grid()
axs1.legend(loc='lower right', fontsize=6.5)
fig1.savefig(os.path.join(save_dir, 'BWBNFD_a.svg'), bbox_inches='tight')

axs2.plot(X_tests_L[i], y_tests_L[i], 'k', linewidth=0.7, label='Reference')
axs2.plot(X_tests_L[i], y_test_preds_L[i], 'b', linewidth=0.7, linestyle='--', alpha=0.7, label='LSTM')
axs2.set_xlabel('Normalized $u$')
axs2.set_ylabel('Normalized $f$')
axs2.grid()
axs2.legend(loc='lower right', fontsize=6)
fig2.savefig(os.path.join(save_dir, 'BWBNFD_d.svg'), bbox_inches='tight')

axs3.plot(X_tests_P[i], y_tests_P[i], 'k', linewidth=0.7, label='Reference')
axs3.plot(X_tests_P[i], y_test_preds_P[i], 'purple', linewidth=0.7, linestyle='--', alpha=0.7, label='PyLSTM')
axs3.set_xlabel('Normalized $u$')
axs3.set_ylabel('Normalized $f$')
axs3.grid()
axs3.legend(loc='lower right', fontsize=5.95)
fig3.savefig(os.path.join(save_dir, 'BWBNFD_g.svg'), bbox_inches='tight')

fig1, axs1 = plt.subplots(1, 1, figsize=(1.8, 1.8))
fig2, axs2 = plt.subplots(1, 1, figsize=(1.8, 1.8))
fig3, axs3 = plt.subplots(1, 1, figsize=(1.8, 1.8))

idc = [1, 8, 10]
i = idc[1]
axs1.plot(X_tests_C[i], y_tests_C[i], 'k', linewidth=0.7, label='Reference')
axs1.plot(X_tests_C[i], y_test_preds_C[i], 'r', linewidth=0.7, linestyle='--', alpha=0.7, label='Proposed')
axs1.set_xlabel('Normalized $u$')
axs1.set_ylabel('Normalized $f$')
axs1.grid()
fig1.savefig(os.path.join(save_dir, 'BWBNFD_b.svg'), bbox_inches='tight')

axs2.plot(X_tests_L[i], y_tests_L[i], 'k', linewidth=0.7, label='Reference')
axs2.plot(X_tests_L[i], y_test_preds_L[i], 'b', linewidth=0.7, linestyle='--', alpha=0.7, label='LSTM')
axs2.set_xlabel('Normalized $u$')
axs2.set_ylabel('Normalized $f$')
axs2.grid()
fig2.savefig(os.path.join(save_dir, 'BWBNFD_e.svg'), bbox_inches='tight')

axs3.plot(X_tests_P[i], y_tests_P[i], 'k', linewidth=0.7, label='Reference')
axs3.plot(X_tests_P[i], y_test_preds_P[i], 'purple', linewidth=0.7, linestyle='--', alpha=0.7, label='PyLSTM')
axs3.set_xlabel('Normalized $u$')
axs3.set_ylabel('Normalized $f$')
axs3.grid()
fig3.savefig(os.path.join(save_dir, 'BWBNFD_h.svg'), bbox_inches='tight')


fig1, axs1 = plt.subplots(1, 1, figsize=(1.8, 1.8))
fig2, axs2 = plt.subplots(1, 1, figsize=(1.8, 1.8))
fig3, axs3 = plt.subplots(1, 1, figsize=(1.8, 1.8))

idc = [1, 8, 10]
i = idc[2]
axs1.plot(X_tests_C[i], y_tests_C[i], 'k', linewidth=0.7, label='Reference')
axs1.plot(X_tests_C[i], y_test_preds_C[i], 'r', linewidth=0.7, linestyle='--', alpha=0.7, label='Proposed')
axs1.set_xlabel('Normalized $u$')
axs1.set_ylabel('Normalized $f$')
axs1.grid()
fig1.savefig(os.path.join(save_dir, 'BWBNFD_c.svg'), bbox_inches='tight')

axs2.plot(X_tests_L[i], y_tests_L[i], 'k', linewidth=0.7, label='Reference')
axs2.plot(X_tests_L[i], y_test_preds_L[i], 'b', linewidth=0.7, linestyle='--', alpha=0.7, label='LSTM')
axs2.set_xlabel('Normalized $u$')
axs2.set_ylabel('Normalized $f$')
axs2.grid()
fig2.savefig(os.path.join(save_dir, 'BWBNFD_f.svg'), bbox_inches='tight')

axs3.plot(X_tests_P[i], y_tests_P[i], 'k', linewidth=0.7, label='Reference')
axs3.plot(X_tests_P[i], y_test_preds_P[i], 'purple', linewidth=0.7, linestyle='--', alpha=0.7, label='PyLSTM')
axs3.set_xlabel('Normalized $u$')
axs3.set_ylabel('Normalized $f$')
axs3.grid()
fig3.savefig(os.path.join(save_dir, 'BWBNFD_i.svg'), bbox_inches='tight')

# axs[0, 0].legend(loc='lower right', fontsize=6.5)
# axs[1, 0].legend(loc='lower right', fontsize=6)
# axs[2, 0].legend(loc='lower right', fontsize=5.95)

# for row in range(0, 2):
#     axs[row, 0].set_xticklabels([])
#     axs[row, 1].set_xticklabels([])
#     axs[row, 2].set_xticklabels([])

# axs[2, 0].set_xlabel('Normalized displacement')
# axs[2, 1].set_xlabel('Normalized displacement')
# axs[2, 2].set_xlabel('Normalized displacement')
# axs[0, 0].set_ylabel('Normalized force')
# axs[1, 0].set_ylabel('Normalized force')
# axs[2, 0].set_ylabel('Normalized force')

# axs[0, 0].text(0.5, -0.05, '(a)', transform=axs[0, 0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[0, 1].text(0.5, -0.05, '(b)', transform=axs[0, 1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[0, 2].text(0.5, -0.05, '(c)', transform=axs[0, 2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1, 0].text(0.5, -0.05, '(d)', transform=axs[1, 0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1, 1].text(0.5, -0.05, '(e)', transform=axs[1, 1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1, 2].text(0.5, -0.05, '(f)', transform=axs[1, 2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2, 0].text(0.5, -0.28, '(g)', transform=axs[2, 0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2, 1].text(0.5, -0.28, '(h)', transform=axs[2, 1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2, 2].text(0.5, -0.28, '(i)', transform=axs[2, 2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
    
# fig.savefig(os.path.join(save_dir, 'BWBNFD.svg'), bbox_inches='tight')





# %% BWBNPK
# Plot test results
Titles = ['BWBN_PE_PI_DA', 'BWBN_PyLSTM', 'BWBN_LSTM']
Data_CustomLSTM = np.load(os.path.join(save_data_dir, Titles[0] + '_Test.npz'))
Data_PyLSTM = np.load(os.path.join(save_data_dir, Titles[1] + '_Test.npz'))
Data_LSTM = np.load(os.path.join(save_data_dir, Titles[2] + '_Test.npz'))

# Plot test results

X_tests_C, y_tests_C, mask_tests_C, y_test_preds_C = Data_CustomLSTM['X_test'], Data_CustomLSTM['y_test'], Data_CustomLSTM['mask_test'], Data_CustomLSTM['y_test_pred']
X_tests_P, y_tests_P, mask_tests_P, y_test_preds_P = Data_PyLSTM['X_test'], Data_PyLSTM['y_test'], Data_PyLSTM['mask_test'], Data_PyLSTM['y_test_pred']
X_tests_L, y_tests_L, mask_tests_L, y_test_preds_L = Data_LSTM['X_test'], Data_LSTM['y_test'], Data_LSTM['mask_test'], Data_LSTM['y_test_pred']

y_tests_C = [y_tests_C[i, mask_tests_C[i].astype(int).astype(bool)] for i in range(len(y_tests_C))]
y_test_preds_C = [y_test_preds_C[i, mask_tests_C[i].astype(int).astype(bool)] for i in range(len(y_test_preds_C))]
y_tests_C = [np.abs(y_test_C).max() for y_test_C in y_tests_C]
y_test_preds_C = [np.abs(y_test_pred_C).max() for y_test_pred_C in y_test_preds_C]
r2_C = r2_score(y_tests_C, y_test_preds_C)

y_tests_P = [y_tests_P[i, mask_tests_P[i].astype(int).astype(bool)] for i in range(len(y_tests_P))]
y_test_preds_P = [y_test_preds_P[i, mask_tests_P[i].astype(int).astype(bool)] for i in range(len(y_test_preds_P))]
y_tests_P = [np.abs(y_test_P).max() for y_test_P in y_tests_P]
y_test_preds_P = [np.abs(y_test_pred_P).max() for y_test_pred_P in y_test_preds_P]
r2_P = r2_score(y_tests_P, y_test_preds_P)

y_tests_L = [y_tests_L[i, mask_tests_L[i].astype(int).astype(bool)] for i in range(len(y_tests_L))]
y_test_preds_L = [y_test_preds_L[i, mask_tests_L[i].astype(int).astype(bool)] for i in range(len(y_test_preds_L))]
y_tests_L = [np.abs(y_test_L).max() for y_test_L in y_tests_L]
y_test_preds_L = [np.abs(y_test_pred_L).max() for y_test_pred_L in y_test_preds_L]
r2_L = r2_score(y_tests_L, y_test_preds_L)

x = np.linspace(0, 1, 100)
fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.scatter(y_tests_C, y_test_preds_C, color='red', s=1, label='Proposed')
axs.text(0.6, 0.1, 'R$^2$ = {:.2f}'.format(r2_C), transform=axs.transAxes, fontsize=6)
axs.set_xlim([0, 0.76])
axs.set_ylim([0, 0.76])
axs.plot(x, x, 'k', linewidth=0.7, label='y=x')
axs.grid()
axs.set_xlabel('True normalized peak')
axs.set_ylabel('Predicted normalized peak')
axs.legend(loc='upper left', fontsize=6)

fig.savefig(os.path.join(save_dir, 'BWBNPK_a.svg'), bbox_inches='tight')


fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.scatter(y_tests_L, y_test_preds_L, color='blue', s=1, label='LSTM')
axs.text(0.6, 0.1, 'R$^2$ = {:.2f}'.format(r2_L), transform=axs.transAxes, fontsize=6)
axs.set_xlim([0, 0.76])
axs.set_ylim([0, 0.76])
axs.plot(x, x, 'k', linewidth=0.7, label='y=x')
axs.grid()
axs.set_xlabel('True normalized peak')
axs.set_ylabel('Predicted normalized peak')
axs.legend(loc='upper left', fontsize=6)

fig.savefig(os.path.join(save_dir, 'BWBNPK_b.svg'), bbox_inches='tight')


fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.scatter(y_tests_P, y_test_preds_P, color='purple', s=1, label='PyLSTM')
axs.text(0.6, 0.1, 'R$^2$ = {:.2f}'.format(r2_P), transform=axs.transAxes, fontsize=6)
axs.set_xlim([0, 0.76])
axs.set_ylim([0, 0.76])
axs.plot(x, x, 'k', linewidth=0.7, label='y=x')
axs.grid()
axs.set_xlabel('True normalized peak')
axs.set_ylabel('Predicted normalized peak')
axs.legend(loc='upper left', fontsize=6)

fig.savefig(os.path.join(save_dir, 'BWBNPK_c.svg'), bbox_inches='tight')

# axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')

# axs[1].scatter(y_tests_L, y_test_preds_L, color='blue', s=1, label='LSTM')
# axs[1].text(0.6, 0.1, 'R$^2$ = {:.2f}'.format(r2_L), transform=axs[1].transAxes, fontsize=6)
# axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2].scatter(y_tests_P, y_test_preds_P, color='purple', s=1, label='PyLSTM')
# axs[2].text(0.6, 0.1, 'R$^2$ = {:.2f}'.format(r2_P), transform=axs[2].transAxes, fontsize=6)
# axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1].set_xlabel('True normalized peak')
# for i in range(3):
#     axs[i].plot(x, x, 'k', linewidth=0.7, label='y=x')
#     axs[i].set_xlim([0, 0.76])
#     axs[i].set_ylim([0, 0.76])
#     axs[i].grid()
#     axs[i].legend(loc='upper left', fontsize=6)

# axs[0].set_ylabel('Predicted normalized peak')
# fig.savefig(os.path.join(save_dir, 'BWBNPK.svg'), bbox_inches='tight')



# %% BWBNPI
Titles = ['BWBN_LSTM', 'BWBN_PE', 'BWBN_PI', 'BWBN_PE_PI_DA']
if generate_data:
    pass
    # Title = Titles[0]
    # print(Title)
    # processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
    # save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
    # model = BasicLSTM(2, 64, 1)
    # test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device, model)
    # Title = Titles[1]
    # print(Title)
    # processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
    # save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
    # test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)
    # Title = Titles[2]
    # print(Title)
    # processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
    # save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
    # model = BasicLSTM(2, 64, 1)
    # test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device, model)
    # Title = Titles[3]
    # print(Title)
    # processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
    # save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
    # test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)
    # # Title = Titles[4]
    # # print(Title)
    # # processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
    # # save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
    # # test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)

    # for Title in Titles:
    #     Data = np.load(os.path.join(save_data_dir, Title + '_Test.npz'))
    #     X_tests, y_tests, mask_tests, y_test_preds, energies_test = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred'][:, :, 0], Data['energies_test'][:, :, 0]
    #     mse_loss = np.sum((y_tests - y_test_preds) ** 2 * mask_tests) / np.sum(mask_tests)
    #     phys_loss = drucker_loss_numpy(y_test_preds, energies_test, mask_tests)
    #     np.savez(os.path.join(save_data_dir, Title + '_Loss.npz'), mse_loss=mse_loss, phys_loss=phys_loss)

mse_losses = []
phys_losses = []
for Title in Titles:
    Data = np.load(os.path.join(save_data_dir, Title + '_Loss.npz'))
    mse_loss = Data['mse_loss']
    phys_loss = Data['phys_loss']
    mse_losses.append(mse_loss)
    phys_losses.append(phys_loss)

total_losses = 0.8 * np.array(mse_losses) + 0.2 * np.array(phys_losses)
rel_losses = total_losses / total_losses[0]
fig, ax = plt.subplots(1, 1, figsize=(4, 1.8))
bars = ax.bar(np.arange(len(Titles[1:])), rel_losses[1:], color='gray')
ax.set_xticks(np.arange(len(Titles[1:])))
ax.set_xticklabels(['PE', 'PI', 'PE, PI, DA'])
ax.set_ylabel('Relative loss')
ax.set_ylim([0, 1.0])
ax.grid(True, axis='y')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, '{:.3f}'.format(yval), ha='center', va='bottom')
fig.savefig(os.path.join(save_dir, 'BWBNPI.svg'), bbox_inches='tight')
# ax.set_yscale('log')

# %% BIFD
Title = 'Bilinear_PE_PI_DA'
# Load best model

processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
if generate_data:
    test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)

# Plot test results
Data = np.load(os.path.join(save_data_dir, Title + '_Test.npz'))

X_tests, y_tests, mask_tests, y_test_preds = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']

X_tests = [X_tests[i, mask_tests[i].astype(int).astype(bool), 0] for i in range(len(X_tests))]
y_tests = [y_tests[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_tests))]
y_test_preds = [y_test_preds[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_test_preds))]

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
idc = [1, 8, 10]
axs.plot(X_tests[idc[0]], y_tests[idc[0]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[0]], y_test_preds[idc[0]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
axs.legend(['Reference', 'Predicted'], loc='lower right', fontsize=6.5)

fig.savefig(os.path.join(save_dir, 'BIFD_a.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.plot(X_tests[idc[1]], y_tests[idc[1]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[1]], y_test_preds[idc[1]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')

fig.savefig(os.path.join(save_dir, 'BIFD_b.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.plot(X_tests[idc[2]], y_tests[idc[2]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[2]], y_test_preds[idc[2]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')

fig.savefig(os.path.join(save_dir, 'BIFD_c.svg'), bbox_inches='tight')
# axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1].set_xlabel('Normalized displacement')
# axs[0].set_ylabel('Normalized force')
# axs[0].legend(['Reference', 'Predicted'], loc='lower right', fontsize=6.5)
# fig.savefig(os.path.join(save_dir, 'BIFD.svg'), bbox_inches='tight')


# %% BITH
Title = 'Bilinear_PE_PI_DA'
Data = np.load(os.path.join(save_data_dir, Title + '_Test.npz'))

X_tests, y_tests, mask_tests, y_test_preds = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']

X_tests = [X_tests[i, mask_tests[i].astype(int).astype(bool), 0] for i in range(len(X_tests))]
y_tests = [y_tests[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_tests))]
y_test_preds = [y_test_preds[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_test_preds))]


# plt.subplots_adjust(hspace=0.4)
idc = [1, 8, 10]
dt = 0.005
t0 = np.arange(0, len(y_tests[idc[0]]) * dt, dt)
t1 = np.arange(0, len(y_tests[idc[1]]) * dt, dt)
t2 = np.arange(0, len(y_tests[idc[2]]) * dt, dt)

fig, axs = plt.subplots(1, 1, figsize=(6, 1.4))
axs.plot(t0, y_tests[idc[0]], 'k', linewidth=0.7, label='Reference')
axs.plot(t0, y_test_preds[idc[0]], 'r', linewidth=0.7, linestyle='--', alpha=0.7, label='Predicted')
axs.set_xlim([t0[0], t0[-1]])
# axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs.grid()
axs.legend(loc='upper left', fontsize=6)
axs.set_xlabel('Time [s]')
axs.set_ylabel('Normalized $f$')

fig.savefig(os.path.join(save_dir, 'BITH_a.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(6, 1.4))
axs.plot(t1, y_tests[idc[1]], 'k', linewidth=0.7)
axs.plot(t1, y_test_preds[idc[1]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.set_xlim([t1[0], t1[-1]])
axs.grid()
axs.set_xlabel('Time [s]')
axs.set_ylabel('Normalized $f$')

fig.savefig(os.path.join(save_dir, 'BITH_b.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(6, 1.4))
axs.plot(t2, y_tests[idc[2]], 'k', linewidth=0.7)
axs.plot(t2, y_test_preds[idc[2]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.set_xlim([t2[0], t2[-1]])
axs.grid()
axs.set_xlabel('Time [s]')
axs.set_ylabel('Normalized $f$')

fig.savefig(os.path.join(save_dir, 'BITH_c.svg'), bbox_inches='tight')
# axs[1].plot(t1, y_tests[idc[1]], 'k', linewidth=0.7)
# axs[1].plot(t1, y_test_preds[idc[1]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
# axs[1].set_xlim([t1[0], t1[-1]])
# axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1].grid()
# axs[2].plot(t2, y_tests[idc[2]], 'k', linewidth=0.7)
# axs[2].plot(t2, y_test_preds[idc[2]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
# axs[2].set_xlim([t2[0], t2[-1]])
# axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2].grid()
# fig.text(0.5, -0.005, 'Time [s]', ha='center', fontsize=8)
# fig.text(0.04, 0.5, 'Normalized force', va='center', rotation='vertical', fontsize=8)
# fig.savefig(os.path.join(save_dir, 'BITH.svg'), bbox_inches='tight')



# %% ROFD
Title = 'RO_PE_PI_DA'
# Load best model

processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')
if generate_data:
    test_prediction(loss_dir, Title, model_dir, processed_data_path, save_data_path, device)

# Plot test results
Data = np.load(os.path.join(save_data_dir, Title + '_Test.npz'))

X_tests, y_tests, mask_tests, y_test_preds = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']

X_tests = [X_tests[i, mask_tests[i].astype(int).astype(bool), 0] for i in range(len(X_tests))]
y_tests = [y_tests[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_tests))]
y_test_preds = [y_test_preds[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_test_preds))]

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
idc = [18, 12, 2]
axs.plot(X_tests[idc[0]], y_tests[idc[0]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[0]], y_test_preds[idc[0]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')
axs.legend(['Reference', 'Predicted'], loc='lower right', fontsize=6.5)

fig.savefig(os.path.join(save_dir, 'ROFD_a.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.plot(X_tests[idc[1]], y_tests[idc[1]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[1]], y_test_preds[idc[1]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')

fig.savefig(os.path.join(save_dir, 'ROFD_b.svg'), bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(1.8, 1.8))
axs.plot(X_tests[idc[2]], y_tests[idc[2]], 'k', linewidth=0.7)
axs.plot(X_tests[idc[2]], y_test_preds[idc[2]], 'r', linewidth=0.7, linestyle='--', alpha=0.7)
axs.grid()
axs.set_xlabel('Normalized $u$')
axs.set_ylabel('Normalized $f$')

fig.savefig(os.path.join(save_dir, 'ROFD_c.svg'), bbox_inches='tight')

# axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
# axs[1].set_xlabel('Normalized displacement')
# axs[0].set_ylabel('Normalized force')
# axs[0].legend(['Reference', 'Predicted'], loc='lower right', fontsize=6.5)
# fig.savefig(os.path.join(save_dir, 'ROFD.svg'), bbox_inches='tight')



# %% RODY
idc = [58, 52, 42]
test_paths = ['ResultAnalysis_for_paper/WCEE2024/Data/RO_PE_PI_DA2_{}.npz'.format(id) for id in idc]
test_datas = [np.load(test_path) for test_path in test_paths]


fig, axs = plt.subplots(3, 1, figsize=(6, 4))
plt.subplots_adjust(hspace=0.5)
for i, test_data in enumerate(test_datas):
    ref_disp = test_data['ref_disp']
    pred_disp = test_data['pred_disp']
    dt = test_data['dt']
    t_ref = np.linspace(0, len(ref_disp) * dt, len(ref_disp))
    t_pred = np.linspace(0, len(pred_disp) * dt, len(pred_disp))
    axs[i].plot(t_ref, ref_disp, 'k', linewidth=0.7)
    axs[i].plot(t_pred, pred_disp, 'r', linewidth=0.7, linestyle='--', alpha=0.7)
    axs[i].set_xlim([0, t_ref[-1]])
    axs[i].grid()
axs[0].text(0.5, -0.25, '(a)', transform=axs[0].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[1].legend(['Reference', 'Predicted'], loc='upper left', fontsize=6.5, ncol=2)
axs[1].text(0.5, -0.25, '(b)', transform=axs[1].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
axs[1].set_ylabel('Displacement [m]')
axs[2].text(0.5, -0.25, '(c)', transform=axs[2].transAxes, fontsize=8.5, fontweight='bold', va='top', ha='center')
fig.text(0.5, -0.01, 'Time [s]', ha='center', fontsize=8)
fig.savefig(os.path.join(save_dir, 'RODY.svg'), bbox_inches='tight')





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




# %%
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = CustomLSTM(2, 64, 1)
for name, param in model.named_parameters():
    print(f"Name: {name}, Shape: {param.shape}")