import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from backend import add_diff, Loss

        
class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear transformation for energy
        self.energy_transform = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # Input gate
        self.fc_i = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)  # *2 for energy
        # Forget gate
        self.fc_f = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Cell state
        self.fc_c = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)
        # Output gate
        self.fc_o = nn.Linear(self.input_dim + self.hidden_dim * 2, self.hidden_dim)

        self.ln_h = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, h_energy, states):
        h, c = states
        h_normalized = self.ln_h(h)
        # Transform the energy
        transformed_energy = torch.tanh(self.energy_transform(h_energy))
        
        h_combined = torch.cat([x, transformed_energy, h_normalized], 1)  # concatenate along the feature dimension

        i = torch.sigmoid((self.fc_i(h_combined)))
        f = torch.sigmoid((self.fc_f(h_combined)))
        g = torch.tanh((self.fc_c(h_combined)))
        o = torch.sigmoid((self.fc_o(h_combined)))
        
        c_next = f * c + i * g
        h_next = self.ln_h(o * torch.tanh(c_next) + h)    # Adding residual connection

        return h_next, c_next


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = CustomLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + 1, output_dim, bias=False)
        

    def forward(self, x, states=None):
        batch_size, seq_length, _ = x.size()

        if states is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            prev_output = torch.zeros(batch_size, 1).to(x.device)
            h_energy = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            energy = torch.zeros(batch_size, 1).to(x.device)
            previous_x = torch.zeros(batch_size, 1).to(x.device)  # Initialize previous_x with zeros
        else:
            h, c, prev_output, h_energy, energy, previous_x = states

        outputs = []
        energies = []
        for t in range(seq_length):
            prev_h = h
            h, c = self.cell(x[:, t, :], h_energy, (h, c))
            output = self.fc(torch.cat([h, x[:, t, 0].unsqueeze(1)], dim=1)) # Need to check if this is correct
            
            # Calculate and accumulate energy using the trapezoid rule
            current_x = x[:, t, 0].unsqueeze(1)
            delta_disp = current_x - previous_x
            h_energy = h_energy + (h + prev_h) / 2 * delta_disp
            energy = energy + (output + prev_output) / 2 * delta_disp
            energies.append(energy)
            prev_output = output
            previous_x = current_x
            outputs.append(output)

        return torch.stack(outputs, dim=1), torch.stack(energies, dim=1), (h, c, prev_output, h_energy, energy, previous_x)

    
def train(X_train,
          y_train,
          mask_train,
          X_val,
          y_val,
          mask_val,
          num_epochs,
          nn_size,
          alpha,
          window_size,
          checkpoint_dir,
          title,
          checkpoint_epoch,
          existing_checkpoint=False,
          augmentation_rate=False,
          result_plot_dir=False):

    if not result_plot_dir:
        result_plot_dir = checkpoint_dir

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    
    model = CustomLSTM(2, nn_size, 1)
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
    outputs, energies, states = outputs, energies, states = model(add_diff(X_train), states)
    LossClass = Loss(outputs[:, :, 0], y_train, energies[:, :, 0], mask_train, device)
    criterion = LossClass.combined_loss

    # Training loop
    model.train()
    epoch_losses = []
    validation_df = pd.DataFrame(columns=['Epoch', 'Loss', 'MSE', 'PhysLoss'])
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
        for step in range(0, X_train.shape[1]-window_size, window_size):
            # print('Step:', step, '//', X_train.shape[1]-window_size, end='\r')
            inputs = X_train[:, step:step+window_size]
            labels = y_train[:, step:step+window_size]

            outputs, energies,  states = model(inputs, states)

            loss = criterion(outputs[:, :, 0], labels, energies[:, :, 0], mask_train[:, step:step+window_size], alpha=alpha)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            states = (states[0].detach(), states[1].detach(), states[2].detach(), states[3].detach(), states[4].detach(), states[5].detach())
            
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
