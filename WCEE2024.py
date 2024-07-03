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
save_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/ResultAnalysis_for_paper/WCEE2024'
os.makedirs(save_dir, exist_ok=True)
generate_data = True
save_data_dir = os.path.join(save_dir, 'Data')
os.makedirs(save_data_dir, exist_ok=True)


EQ_data_dir = '/home/jaehwan/Python Project/DLCM/Data'
hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis'
processed_data_dir = hysteresis_data_dir
model_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Models'
loss_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Result Plots'





# %% Bouc-Wen dynamic results
Title = 'BoucWen_PE_PI_DA2'
EQ_number = 0
if generate_data:

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

    Data = np.load(os.path.join(hysteresis_data_dir, Title + '_{}.npz'.format(EQ_number)))
    disp = Data['disp']
    force = Data['force']
    os.makedirs(hysteresis_data_dir, exist_ok=True)

    """Changing the EQ.AT2 file to .npy file"""
    dt_list, nPts_list = [], []
    for EQ_name in EQ_list:
        file_name, file_extension = os.path.splitext(EQ_name)
        dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
        dt_list.append(dt)
        nPts_list.append(nPts)

    EQ_name = EQ_list[EQ_number]
    file_name, file_extension = os.path.splitext(EQ_name)
    gm = torch.tensor(np.load(file_name+'.npy'))
    dt = dt_list[EQ_number]
    X, F = model.dynamic_analysis(dt, -gm, 1, 0, device=device)
    X, F = X.cpu().detach().numpy(), F.cpu().detach().numpy()
    np.savez(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)), pred_disp=X[0, :-1, 0], pred_force=F[:-1], ref_disp=disp[::2], ref_force=force[::2], dt=dt)





# %%
Data = np.load(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)))
pred_disp, pred_force, ref_disp, ref_force, dt = Data['pred_disp'], Data['pred_force'], Data['ref_disp'], Data['ref_force'], Data['dt']
pred_t = np.linspace(0, (len(pred_disp)-1)*dt, num=len(pred_disp))
ref_t = np.linspace(0, (len(ref_disp)-1)*dt, num=len(ref_disp))

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.plot(ref_t, ref_disp, label='Bouc-Wen', color='dimgray', alpha=0.3)
ax.plot(pred_t, pred_disp, label='Predicted', color='blue', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m)')
ax.legend()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'BoucWen_dynamic_disp.svg')), bbox_inches='tight')





# %%
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
ax.plot(disp[::2], force[::2], label='Bouc-Wen', color='dimgray', alpha=0.3)
ax.plot(X[0, :-1, 0], F[:-1], label='Predicted', color='blue', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Displacement (m)')
ax.set_ylabel('Force (kN)')
ax.legend()
# ax.grid()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'BoucWen_dynamic_hysteresis.svg')), bbox_inches='tight')
# %%











# %% BWBN dynamic results
Title = 'BWBN_PE_PI_DA2'
EQ_number = 1
if generate_data:

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

    Data = np.load(os.path.join(hysteresis_data_dir, Title + '_{}.npz'.format(EQ_number)))
    disp = Data['disp']
    force = Data['force']
    os.makedirs(hysteresis_data_dir, exist_ok=True)

    """Changing the EQ.AT2 file to .npy file"""
    dt_list, nPts_list = [], []
    for EQ_name in EQ_list:
        file_name, file_extension = os.path.splitext(EQ_name)
        dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
        dt_list.append(dt)
        nPts_list.append(nPts)

    EQ_name = EQ_list[EQ_number]
    file_name, file_extension = os.path.splitext(EQ_name)
    gm = torch.tensor(np.load(file_name+'.npy'))
    dt = dt_list[EQ_number]
    X, F = model.dynamic_analysis(dt, -gm, 1, 0, device=device)
    X, F = X.cpu().detach().numpy(), F.cpu().detach().numpy()
    np.savez(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)), pred_disp=X[0, :-1, 0], pred_force=F[:-1], ref_disp=disp[::2], ref_force=force[::2], dt=dt)





# %%
Data = np.load(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)))
pred_disp, pred_force, ref_disp, ref_force, dt = Data['pred_disp'], Data['pred_force'], Data['ref_disp'], Data['ref_force'], Data['dt']
pred_t = np.linspace(0, (len(pred_disp)-1)*dt, num=len(pred_disp))
ref_t = np.linspace(0, (len(ref_disp)-1)*dt, num=len(ref_disp))

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.plot(ref_t, ref_disp, label='BWBN', color='dimgray', alpha=0.3)
ax.plot(pred_t, pred_disp, label='Predicted', color='blue', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m)')
ax.legend()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'BWBN_dynamic_disp.svg')), bbox_inches='tight')





# %%
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
ax.plot(disp[::2], force[::2], label='BWBN', color='dimgray', alpha=0.3)
ax.plot(X[0, :-1, 0], F[:-1], label='Predicted', color='blue', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Displacement (m)')
ax.set_ylabel('Force (kN)')
ax.legend()
# ax.grid()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'BWBN_dynamic_hysteresis.svg')), bbox_inches='tight')
# %%







# %% RO dynamic results
Title = 'RO_PE_PI_DA2'
EQ_number = 11
if generate_data:

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

    Data = np.load(os.path.join(hysteresis_data_dir, Title + '_{}.npz'.format(EQ_number)))
    disp = Data['disp']
    force = Data['force']
    os.makedirs(hysteresis_data_dir, exist_ok=True)

    """Changing the EQ.AT2 file to .npy file"""
    dt_list, nPts_list = [], []
    for EQ_name in EQ_list:
        file_name, file_extension = os.path.splitext(EQ_name)
        dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
        dt_list.append(dt)
        nPts_list.append(nPts)

    EQ_name = EQ_list[EQ_number]
    file_name, file_extension = os.path.splitext(EQ_name)
    gm = torch.tensor(np.load(file_name+'.npy'))
    dt = dt_list[EQ_number]
    X, F = model.dynamic_analysis(dt, -gm, 1, 0, device=device)
    X, F = X.cpu().detach().numpy(), F.cpu().detach().numpy()
    np.savez(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)), pred_disp=X[0, :-1, 0], pred_force=F[:-1], ref_disp=disp[::2], ref_force=force[::2], dt=dt)





# %%
Data = np.load(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)))
pred_disp, pred_force, ref_disp, ref_force, dt = Data['pred_disp'], Data['pred_force'], Data['ref_disp'], Data['ref_force'], Data['dt']
pred_t = np.linspace(0, (len(pred_disp)-1)*dt, num=len(pred_disp))
ref_t = np.linspace(0, (len(ref_disp)-1)*dt, num=len(ref_disp))

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.plot(ref_t, ref_disp, label='Ramberg-Osgood', color='dimgray', alpha=0.3)
ax.plot(pred_t, pred_disp, label='Predicted', color='red', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m)')
ax.legend()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'RO_dynamic_disp.svg')), bbox_inches='tight')





# %%
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
ax.plot(disp[::2], force[::2], label='Ramberg-Osgood', color='dimgray', alpha=0.3)
ax.plot(X[0, :-1, 0], F[:-1], label='Predicted', color='red', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Displacement (m)')
ax.set_ylabel('Force (kN)')
ax.legend()
# ax.grid()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'RO_dynamic_hysteresis.svg')), bbox_inches='tight')
# %%






# %% Bilinear dynamic results
Title = 'Bilinear_PE_PI_DA2'
EQ_number = 0
if generate_data:

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

    Data = np.load(os.path.join(hysteresis_data_dir, Title + '_{}.npz'.format(EQ_number)))
    disp = Data['disp']
    force = Data['force']
    os.makedirs(hysteresis_data_dir, exist_ok=True)

    """Changing the EQ.AT2 file to .npy file"""
    dt_list, nPts_list = [], []
    for EQ_name in EQ_list:
        file_name, file_extension = os.path.splitext(EQ_name)
        dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
        dt_list.append(dt)
        nPts_list.append(nPts)

    EQ_name = EQ_list[EQ_number]
    file_name, file_extension = os.path.splitext(EQ_name)
    gm = torch.tensor(np.load(file_name+'.npy'))
    dt = dt_list[EQ_number]
    X, F = model.dynamic_analysis(dt, -gm, 1, 0, device=device)
    X, F = X.cpu().detach().numpy(), F.cpu().detach().numpy()
    np.savez(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)), pred_disp=X[0, :-1, 0], pred_force=F[:-1], ref_disp=disp[::2], ref_force=force[::2], dt=dt)





# %%
Data = np.load(os.path.join(save_data_dir, Title + '_{}.npz'.format(EQ_number)))
pred_disp, pred_force, ref_disp, ref_force, dt = Data['pred_disp'], Data['pred_force'], Data['ref_disp'], Data['ref_force'], Data['dt']
pred_t = np.linspace(0, (len(pred_disp)-1)*dt, num=len(pred_disp))
ref_t = np.linspace(0, (len(ref_disp)-1)*dt, num=len(ref_disp))

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.plot(ref_t, ref_disp, label='Bilinear', color='dimgray', alpha=0.3)
ax.plot(pred_t, pred_disp, label='Predicted', color='red', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement (m)')
ax.legend()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'Bilinear_dynamic_disp.svg')), bbox_inches='tight')





# %%
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
ax.plot(disp[::2], force[::2], label='Bilinear', color='dimgray', alpha=0.3)
ax.plot(X[0, :-1, 0], F[:-1], label='Predicted', color='red', alpha=0.7, linewidth=0.5, linestyle='--')
ax.set_xlabel('Displacement (m)')
ax.set_ylabel('Force (kN)')
ax.legend()
# ax.grid()
fig.savefig(os.path.normpath(os.path.join(save_dir, 'Bilinear_dynamic_hysteresis.svg')), bbox_inches='tight')





# %% Create hysteresis
EQ_numbers = [0, 1, 2]
Title = 'Bilinear_PE_PI_DA2'
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

for EQ_number in EQ_numbers:

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax.plot(X_tests[EQ_number], y_tests[EQ_number], label='Bilinear', color='dimgray', alpha=0.3)
    ax.set_xlabel('Displacement (m)')
    ax.set_ylabel('Force (kN)')
    fig.savefig(os.path.normpath(os.path.join(save_dir, 'Bilinear_sample_hysteresis_{}.svg'.format(EQ_number))), bbox_inches='tight')
    ax.plot(X_tests[EQ_number], y_test_preds[EQ_number], label='Predicted', color='red', alpha=0.7, linewidth=0.5, linestyle='--')
    fig.savefig(os.path.normpath(os.path.join(save_dir, 'Bilinear_sample_predicted_hysteresis_{}.svg'.format(EQ_number))), bbox_inches='tight')
    plt.close(fig)
# %%
# %% Create hysteresis
EQ_numbers = [0, 1, 2]
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

for EQ_number in EQ_numbers:

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    ax.plot(X_tests[EQ_number], y_tests[EQ_number], label='Bouc-Wen', color='dimgray', alpha=0.3)
    ax.set_xlabel('Displacement (m)')
    ax.set_ylabel('Force (kN)')
    fig.savefig(os.path.normpath(os.path.join(save_dir, 'BoucWen_sample_hysteresis_{}.svg'.format(EQ_number))), bbox_inches='tight')
    ax.plot(X_tests[EQ_number], y_test_preds[EQ_number], label='Predicted', color='red', alpha=0.7, linewidth=0.5, linestyle='--')
    fig.savefig(os.path.normpath(os.path.join(save_dir, 'BoucWen_sample_predicted_hysteresis_{}.svg'.format(EQ_number))), bbox_inches='tight')
    plt.close(fig)
