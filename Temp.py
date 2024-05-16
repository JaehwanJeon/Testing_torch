# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import ReadRecord
from backend import *
import Experiment_230508 as Experiment

plt.rcParams['font.size'] = 7 # 기본 글씨 크기 설정
plt.rcParams['font.family'] = 'serif' # 글씨체 기본 설정
plt.rcParams['axes.titlesize'] = 7 # 제목의 기본 글씨 크기 설정
plt.rcParams['axes.labelsize'] = 8 # 축 라벨의 기본 글씨 크기 설정
plt.rcParams['xtick.labelsize'] = 7 # x축 눈금 라벨의 기본 글씨 크기 설정
plt.rcParams['ytick.labelsize'] = 7 # y축 눈금 라벨의 기본 글씨 크기 설정
plt.rcParams['legend.fontsize'] = 7 # 범례의 기본 글씨 크기 설정


device = torch.device('cuda:0')
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


Title = 'BoucWen_PE_PI_DA2'
processed_data_path = os.path.normpath(os.path.join(processed_data_dir, './' + Title + '_Processed_data.npz'))
save_data_path = os.path.join(save_data_dir, Title + '_Test.npz')

if generate_data:
    test_prediction_denormalize(loss_dir,
                    Title,
                    model_dir,
                    processed_data_path,
                    save_data_path,
                    device)


Data = np.load(os.path.join(save_data_dir, Title + '_Test.npz'))

X_tests, y_tests, mask_tests, y_test_preds = Data['X_test'], Data['y_test'], Data['mask_test'], Data['y_test_pred']

X_tests = [X_tests[i, mask_tests[i].astype(int).astype(bool), 0] for i in range(len(X_tests))]
y_tests = [y_tests[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_tests))]
y_test_preds = [y_test_preds[i, mask_tests[i].astype(int).astype(bool)] for i in range(len(y_test_preds))]

with open('EQ_list.txt', 'r') as f:
    EQ_list_ = [line.strip() for line in f]
print(EQ_list_[:3])

EQ_list = []
for i, EQ in enumerate(EQ_list_):
    EQ_name = './'+EQ
    EQ_list.append(os.path.normpath(os.path.join('/home/jaehwan/Python Project/DLCM/Data', EQ_name)).replace("\\", "/"))
print(EQ_list[:3])





loss_data = pd.read_csv(os.path.normpath(os.path.join(loss_dir, Title + '_loss.csv')))
losses = loss_data['Loss'].values

best_model_idx = np.argmin(losses)
best_model_epoch = loss_data['Epoch'].values[best_model_idx]
checkpoint = torch.load(os.path.normpath(os.path.join(model_dir, './' + Title +'_checkpoint_{}.pth'.format(int(best_model_epoch)))))

nn_size = checkpoint['cell.fc_f.bias'].size()[0]
model = CustomLSTM(2, nn_size, 1)
model.load_state_dict(checkpoint)
model.load_norm_factors()
model = model.to(device)
model.eval()





#%%
i = 0

Data = np.load(os.path.join(hysteresis_data_dir, Title + '_{}.npz'.format(i)))
disp = Data['disp']
os.makedirs(hysteresis_data_dir, exist_ok=True)

"""Changing the EQ.AT2 file to .npy file"""
dt_list, nPts_list = [], []
for EQ_name in EQ_list:
    file_name, file_extension = os.path.splitext(EQ_name)
    dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
    dt_list.append(dt)
    nPts_list.append(nPts)

EQ_name = EQ_list[i]
file_name, file_extension = os.path.splitext(EQ_name)
gm = torch.tensor(np.load(file_name+'.npy'))
dt = dt_list[i]
X = model.dynamic_analysis(dt, -gm, 1, 0, device=device)
X = X.cpu().detach().numpy()
# %%

plt.plot(disp[::2], label='Original')
plt.plot(X[0, :, 0], label='Predicted')
# %%
