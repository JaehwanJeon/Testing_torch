import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import Training
import re
import pandas as pd
from backend import add_diff, Loss

def test_impact(X_test,
                y_test,
                mask_test,
                impact_length_list,
                impact_magnitude_list,
                model_paths,
                title,
                nn_size,
                result_plot_dir):
    raise NotImplementedError("This part is not implemented yet.")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')

    loss_data_file = os.path.normpath(os.path.join(result_plot_dir, './' + title + '_loss.csv'))
    losses = pd.read_csv(loss_data_file)['Loss'].values

    best_model_idx = np.argmin(losses)
    checkpoint = torch.load(model_paths[best_model_idx])
    model = Training.CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    X_test, y_test, mask_test = torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).float().to(device), torch.from_numpy(mask_test).float().to(device)

    y_test_pred, energies_test_pred, _ = model(X_test)
    
    mse_list = []
    drucker_list = []
    for i in range(len(y_test_pred)):
        mse_list.append(Training.custom_loss(y_test_pred[i, :, 0], y_test[i, :], mask_test[i, :]).cpu().detach().numpy().item())
        drucker_list.append(Training.drucker_loss(y_test_pred[i:i+1, :, 0], energies_test_pred[i:i+1, :, 0], mask_test[i:i+1, :]).cpu().detach().numpy().item())
    y_test_pred = y_test_pred.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    mask_test = mask_test.cpu().detach().numpy()

    i = 0
    for impact_length in impact_length_list:
        for impact_magnitude in impact_magnitude_list:
            X_test_i = X_test[i, mask_test[i].astype(int).astype(bool), 0]
            y_test_pred_i = y_test_pred[i, mask_test[i].astype(int).astype(bool)]
            y_test_i = y_test[i, mask_test[i].astype(int).astype(bool)]
            plt.figure(figsize=(8, 8))
            plt.plot(X_test_i, y_test_i, 'b')
            plt.plot(X_test_i, y_test_pred_i, 'r')
            plt.text(0.8, 0.3, 'MSE: {:.4e}\nDrucker: {:.4e}'.format(mse_list[i], drucker_list[i]), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
            plt.grid()
            plt.xlabel('Displacement (m)', fontsize=12)
            plt.ylabel('Force (N)', fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.legend(['Reference', 'Predicted'], fontsize=12)
            plt.savefig(os.path.normpath(os.path.join(result_plot_dir, './' + title + '_{}_{}_test.png'.format(impact_length, impact_magnitude))))
            plt.close()
            i += 1


def result_plot(X_val,
                y_val,
                mask_val,
                X_test,
                y_test,
                mask_test,
                nn_size,
                model_dir,
                model_paths,
                title,
                result_plot_dir):

    os.makedirs(result_plot_dir, exist_ok=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    X_val, X_test = torch.from_numpy(X_val).float().to(device), torch.from_numpy(X_test).float().to(device)
    X_val, X_test = add_diff(X_val), add_diff(X_test)
    y_val, y_test = torch.from_numpy(y_val).float().to(device), torch.from_numpy(y_test).float().to(device)
    mask_val, mask_test = torch.from_numpy(mask_val).float().to(device), torch.from_numpy(mask_test).float().to(device)

    epochs = []
    losses = []
    weights = []
    loss_data_file = os.path.normpath(os.path.join(result_plot_dir, './' + title + '_loss.csv'))
    
    if not os.path.exists(loss_data_file):

        for model_path in model_paths:
            match = re.search('_checkpoint_(\d+).pth', model_path)
            if match:
                epoch =  int(match.group(1))
            checkpoint = torch.load(model_path)
            model = Training.CustomLSTM(2, nn_size, 1)
            model.load_state_dict(checkpoint)
            weight = model.cell.energy_transform.weight.cpu().detach().numpy()
            model = model.to(device)
            model.eval()
            y_val_pred, energies_val, _ = model(X_val)

            loss = Training.custom_loss(y_val_pred[:, :, 0], y_val, mask_val)
            losses.append(loss.item())
            epochs.append(epoch)
            weights.append(weight)
            print('Validation Loss: {}'.format(loss.item()))
        
        pd.DataFrame({'Epoch': epochs, 'Loss': losses, 'Weight': weights}).to_csv(os.path.normpath(os.path.join(result_plot_dir, './' + title + '_loss.csv')), index=False)

    loss_data = pd.read_csv(loss_data_file)
    losses = loss_data['Loss'].values

    best_model_idx = np.argmin(losses)
    best_model_epoch = loss_data['Epoch'].values[best_model_idx]
    checkpoint = torch.load(os.path.normpath(os.path.join(model_dir, './' + title +'_checkpoint_{}.pth'.format(int(best_model_epoch)))))
    model = Training.CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    y_val_pred, energies_val, _ = model(X_val)
    y_val_pred = y_val_pred.cpu().detach().numpy()
    X_val = X_val.cpu().detach().numpy()
    y_val = y_val.cpu().detach().numpy()
    mask_val = mask_val.cpu().detach().numpy()

    for i in range(len(y_val_pred)):
        X_val_i = X_val[i, mask_val[i].astype(int).astype(bool), 0]
        y_val_pred_i = y_val_pred[i, mask_val[i].astype(int).astype(bool)]
        y_val_i = y_val[i, mask_val[i].astype(int).astype(bool)]
        plt.figure(figsize=(8, 8))
        plt.plot(X_val_i, y_val_i, 'b')
        plt.plot(X_val_i, y_val_pred_i, 'r')
        plt.grid()
        plt.xlabel('Displacement (m)', fontsize=12)
        plt.ylabel('Force (N)', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(['Reference', 'Predicted'], fontsize=12)
        plt.savefig(os.path.normpath(os.path.join(result_plot_dir, './' + title + '_{}_val.png'.format(i))))
        plt.close()
