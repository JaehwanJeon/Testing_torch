import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from backend import *


def train(X_train,
          y_train,
          mask_train,
          X_val,
          y_val,
          mask_val,
          model,
          num_epochs,
          alpha,
          window_size,
          checkpoint_dir,
          title,
          checkpoint_epoch,
          existing_checkpoint=False,
          augmentation_rate=False,
          result_plot_dir=False,
          batch_size=False,
          device = torch.device('cuda:0')):

    if not result_plot_dir:
        result_plot_dir = checkpoint_dir

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if existing_checkpoint != False:
        model.load_state_dict(existing_checkpoint)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    X_train_, y_train_, mask_train_ = torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device), torch.from_numpy(mask_train).float().to(device)
    X_train, y_train, mask_train = X_train_, y_train_, mask_train_
    X_val, y_val, mask_val = torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).float().to(device), torch.from_numpy(mask_val).float().to(device)
    X_val = add_diff(X_val)
    val_loss = []
    # Obtaining initial normalization factor for data-driven loss vs physics-based loss
    states = None
    model.eval()
    outputs, energies, states = model(add_diff(X_train), states)
    if batch_size != False:
        batch_idx = torch.randperm(X_train.shape[0])[:batch_size]
        LossClass = Loss(outputs[batch_idx, :, 0], y_train[batch_idx], energies[batch_idx, :, 0], mask_train[batch_idx], device)
    else:
        LossClass = Loss(outputs[:, :, 0], y_train, energies[:, :, 0], mask_train, device)
    criterion = LossClass.combined_loss

    # Training loop
    model.train()
    epoch_losses = []
    validation_df = pd.DataFrame(columns=['Epoch', 'Loss', 'MSE', 'PhysLoss'])
    if batch_size != False:
        for epoch in range(num_epochs):
            losses = []
            print('Epoch:', epoch+1)
            for STEP in range(X_train_.shape[0]//batch_size + 1):
                if augmentation_rate != False:
                    augmented_idx = torch.randperm(X_train_.shape[1])[:int(X_train_.shape[1] * augmentation_rate)].to(device)
                    augmented_idx, _ = torch.sort(augmented_idx)
                    X_train, y_train, mask_train = X_train_[:, augmented_idx], y_train_[:, augmented_idx], mask_train_[:, augmented_idx]
                    X_train = add_diff(X_train)
                batch_idx = torch.randperm(X_train.shape[0])[:batch_size]
                states = None
                for step in range(0, X_train.shape[1]-int(window_size) + 1, int(window_size)):
                    # print('Step:', step, '//', X_train.shape[1]-window_size, end='\r')
                    inputs = X_train[batch_idx, step:step+window_size]
                    labels = y_train[batch_idx, step:step+window_size]

                    outputs, energies,  states = model(inputs, states)

                    loss = criterion(outputs[:, :, 0], labels, energies[:, :, 0], mask_train[batch_idx, step:step+window_size], alpha=alpha)
                    losses.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    states = tuple(state.detach() for state in states)
                
            avg_loss = sum(losses) / len(losses)
            epoch_losses.append(avg_loss)
            print('\n', f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')        

            # 일정 주기마다 손실 출력
            if (epoch + 1) % checkpoint_epoch == 0:
                outputs, energies,  _ = model(X_val)
                loss = criterion(outputs[:, :, 0], y_val, energies[:, :, 0], mask_val, alpha=alpha)
                new_row = {'Epoch': epoch+1, 'Loss': loss.item(), 'MSE': LossClass.mse_loss.item(), 'PhysLoss': LossClass.phys_loss.item(), 'TrainLoss': avg_loss}
                validation_df = validation_df.append(new_row, ignore_index=True)
                validation_df.to_csv(os.path.normpath(os.path.join(result_plot_dir, title + '_loss.csv')), index=False)
                val_loss.append(loss.item())
                print(f'Validation Loss: {loss.item()}')
                torch.save(model.state_dict(), os.path.normpath(os.path.join(checkpoint_dir, title + '_checkpoint_{}.pth'.format(epoch+1))))

    else:
        for epoch in range(num_epochs):
            if augmentation_rate != False:
                augmented_idx = np.random.choice(X_train_.shape[1], int(X_train_.shape[1] * augmentation_rate), replace=False)
                augmented_idx = np.sort(augmented_idx)
                X_train, y_train, mask_train = X_train_[:, augmented_idx], y_train_[:, augmented_idx], mask_train_[:, augmented_idx]
                X_train = add_diff(X_train)
            else:
                X_train = add_diff(X_train_)
            states = None
            losses = []
            print('Epoch:', epoch+1)
            for step in range(0, X_train.shape[1]- window_size, window_size):
                # print('Step:', step, '//', X_train.shape[1]-window_size, end='\r')
                inputs = X_train[:, step:step+window_size]
                labels = y_train[:, step:step+window_size]

                outputs, energies,  states = model(inputs, states)

                loss = criterion(outputs[:, :, 0], labels, energies[:, :, 0], mask_train[:, step:step+window_size], alpha=alpha)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                states = tuple(state.detach() for state in states)
                
            avg_loss = sum(losses) / len(losses)
            epoch_losses.append(avg_loss)
            print('\n', f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')        

            # 일정 주기마다 손실 출력
            if (epoch + 1) % checkpoint_epoch == 0:
                outputs, energies,  _ = model(X_val)
                loss = criterion(outputs[:, :, 0], y_val, energies[:, :, 0], mask_val, alpha=alpha)
                new_row = {'Epoch': epoch+1, 'Loss': loss.item(), 'MSE': LossClass.mse_loss.item(), 'PhysLoss': LossClass.phys_loss.item()}
                validation_df = validation_df.append(new_row, ignore_index=True)
                validation_df.to_csv(os.path.normpath(os.path.join(result_plot_dir, title + '_loss.csv')), index=False)
                val_loss.append(loss.item())
                print(f'Validation Loss: {loss.item()}')
                torch.save(model.state_dict(), os.path.normpath(os.path.join(checkpoint_dir, title + '_checkpoint_{}.pth'.format(epoch+1))))
            
    np.savez(os.path.normpath(os.path.join(checkpoint_dir, title + '_losses.npz')), train_loss=epoch_losses, val_loss=val_loss)
    return model
