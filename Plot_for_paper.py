#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import ReadRecord


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
generate_data = True

Title = 'BoucWen_PE_PI_DA'

EQ_data_dir = '/home/jaehwan/Python Project/DLCM/Data'
hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis'




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





# %%
