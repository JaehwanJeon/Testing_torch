import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import Training

def result_plot(X_val,
                y_val,
                mask_val,
                X_test,
                y_test,
                mask_test,
                nn_size,
                model_paths,
                result_plot_dir):

    os.makedirs(result_plot_dir, exist_ok=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1')
    X_val, X_test = torch.from_numpy(X_val).float().to(device), torch.from_numpy(X_test).float().to(device)
    y_val, y_test = torch.from_numpy(y_val).float().to(device), torch.from_numpy(y_test).float().to(device)
    mask_val, mask_test = torch.from_numpy(mask_val).float().to(device), torch.from_numpy(mask_test).float().to(device)
    
    losses = []
    for model_path in model_paths:
        checkpoint = torch.load(model_path)
        model = Training.CustomLSTM(2, nn_size, 1)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        y_val_pred, _ = model(X_val)
        loss = Training.custom_loss(y_val_pred[:, :, 0], y_val, mask_val)
        losses.append(loss.item())
        print('Validation Loss: {}'.format(loss.item()))
    
    best_model_idx = np.argmin(losses)
    checkpoint = torch.load(model_paths[best_model_idx])
    model = Training.CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    y_val_pred, _ = model(X_val)
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
        plt.savefig(os.path.normpath(os.path.join(result_plot_dir, './{}_val.png'.format(i))))
        plt.close()
