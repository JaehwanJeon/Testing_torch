"""
If successful, then add model_name='PINN_{}_{}_{}'.format(num_inputs, nn_size, 40)'
Should give random.seed
Any ways to save history?
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

        
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

    

def custom_loss(y_pred, y, mask):
    loss = torch.mean((y_pred - y)**2 * mask)
    return loss

def drucker_loss(f, e, mask):
    
    num_samples, time_length = f.size()
    num_chops = 201
    chop_vector = torch.linspace(-1, 1, num_chops).to(f.device)

    # Expand the mask and tensors along the second axis
    mask_expanded = mask.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
    f_expanded = f.unsqueeze(1).expand(-1, chop_vector.size(0), -1)
    e_expanded = e.unsqueeze(1).expand(-1, chop_vector.size(0), -1)

    chop_vector_expanded = chop_vector.unsqueeze(0).unsqueeze(-1).expand(num_samples, -1, time_length)
    deducted_f_expanded = f_expanded - chop_vector_expanded

    # Create deducted_f_sign tensor with size [num_samples, num_chops, time_length]
    deducted_f_sign = (deducted_f_expanded > 0).int()

    diff_sign = torch.diff(deducted_f_sign, dim=2, prepend=deducted_f_sign[:, :, 0].unsqueeze(2))
    change_bool = ((diff_sign != 0) * mask_expanded).bool()
    change_idx = torch.nonzero(change_bool)
    selected_e = e_expanded[change_idx[:, 0], change_idx[:, 1], change_idx[:, 2]]
    diff_e = selected_e[1:] - selected_e[:-1]
    diff_idx = change_idx[1:] - change_idx[:-1]
    diff_e = diff_e * (diff_idx[:, 0] == 0) * (diff_idx[:, 1] == 0)
    diff_e_neg = diff_e[diff_e < 0]
    return -torch.sum(diff_e_neg) / (num_samples * time_length)
    


class Loss:
    def __init__(self, y_pred, y, energies, mask, device):
        self.mse_loss = (torch.mean((y_pred - y)**2 * mask)).to(device).detach()
        self.phys_loss = (drucker_loss(y_pred, energies, mask)).to(device).detach()
        print('MSE Loss / Phys Loss will be normalized by {}/{}'.format(self.mse_loss, self.phys_loss))
        pass

    def combined_loss(self, y_pred, y, energies, mask, alpha=0.2):
        mse_loss = torch.mean((y_pred - y)**2 * mask) / self.mse_loss
        phys_loss = drucker_loss(y_pred, energies, mask) / self.phys_loss  # Assuming y_pred and y are 2D tensors [batch_size x seq_length]
        print('MSE Loss / Phys Loss: {:.8f}/{:.8f}'.format(mse_loss, phys_loss), end='\r')
        return (1-alpha) * mse_loss + alpha * phys_loss

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
          existing_checkpoint=False):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    
    model = CustomLSTM(2, nn_size, 1)
    if existing_checkpoint != False:
        model.load_state_dict(existing_checkpoint)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_train, y_train, mask_train = torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device), torch.from_numpy(mask_train).float().to(device)
    X_val, y_val, mask_val = torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).float().to(device), torch.from_numpy(mask_val).float().to(device)

    val_loss = []
    # Obtaining initial normalization factor for data-driven loss vs physics-based loss
    states = None
    model.eval()
    outputs, energies, states = model(X_train, states)
    criterion = Loss(outputs[:, :, 0], y_train, energies[:, :, 0], mask_train, device).combined_loss

    # Training loop
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        states = None
        losses = []
        print('Epoch:', epoch+1)
        for step in range(0, X_train.shape[1]-window_size, window_size):
            # print('Step:', step, '//', X_train.shape[1]-window_size, end='\r')
            inputs = X_train[:, step:step+window_size]
            labels = y_train[:, step:step+window_size]

            outputs, energies,  states = model(inputs, states)

            # # For debugging
            # plt.plot(energies[0, :, 0].cpu().detach().numpy())
            # plt.savefig(os.path.normpath(os.path.join(checkpoint_dir, './Temp/' + title + '_energy_training_temp.png')))
            # plt.close()
            # #

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
            loss = custom_loss(outputs[:, :, 0], y_val, mask_val)
            val_loss.append(loss.item())
            print(f'Validation Loss: {loss.item()}')
            torch.save(model.state_dict(), os.path.normpath(os.path.join(checkpoint_dir, title + '_checkpoint_{}.pth'.format(epoch+1))))
        
    np.savez(os.path.normpath(os.path.join(checkpoint_dir, title + '_losses.npz')), train_loss=epoch_losses, val_loss=val_loss)
    return model



