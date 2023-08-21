import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import BWBNTesting_Training

# def result_plot(range_plot,
#                 result_plot_dir,
#                 show_plot,
#                 model_path,
#                 hysteresis_data_dir,
#                 num_inputs,
#                 output_factor):
#
#     os.makedirs(result_plot_dir, exist_ok=True)
#     DNN_model = keras.models.load_model(model_path)
#
#     for i in range(range_plot[0], range_plot[1]):
#         data_path = os.path.normpath(os.path.join(hysteresis_data_dir, './{}.npz'.format(i)))
#         save_path = os.path.normpath(os.path.join(result_plot_dir, './{}.png'.format(i)))
#         test_plot(data_path, DNN_model, num_inputs, output_factor, save_path=save_path, show_plot=show_plot)


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
        model = BWBNTesting_Training.CustomLSTM(2, nn_size, 1)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        y_val_pred, _ = model(X_val)
        loss = BWBNTesting_Training.custom_loss(y_val_pred[:, :, 0], y_val, mask_val)
        losses.append(loss.item())
        print('Validation Loss: {}'.format(loss.item()))
    
    best_model_idx = np.argmin(losses)
    checkpoint = torch.load(model_paths[best_model_idx])
    model = BWBNTesting_Training.CustomLSTM(2, nn_size, 1)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    y_val_pred, _ = model(X_val)
    y_val_pred = y_val_pred.cpu().detach().numpy()
    X_val = X_val.cpu().detach().numpy()
    y_val = y_val.cpu().detach().numpy()
    mask_val = mask_val.cpu().detach().numpy()

    for i in range(len(y_val_pred)):
        X_val_i = X_val[i, mask_val[i].astype(bool)]
        y_val_pred_i = y_val_pred[i, mask_val[i].astype(bool)]
        y_val_i = y_val[i, mask_val[i].astype(bool)]
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
