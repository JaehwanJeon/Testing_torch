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
        
        # Input gate
        self.fc_i = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.ln_i = nn.LayerNorm(self.hidden_dim)
        # Forget gate
        self.fc_f = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.ln_f = nn.LayerNorm(self.hidden_dim)
        # Cell state
        self.fc_c = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.ln_c = nn.LayerNorm(self.hidden_dim)
        # Output gate
        self.fc_o = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.ln_o = nn.LayerNorm(self.hidden_dim)

        self.ln_h = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, states):
        h, c = states
        
        combined = torch.cat([x, h], 1)  # concatenate along the feature dimension

        i = torch.sigmoid(self.ln_i(self.fc_i(combined)))
        f = torch.sigmoid(self.ln_f(self.fc_f(combined)))
        g = torch.tanh(self.ln_c(self.fc_c(combined)))
        o = torch.sigmoid(self.ln_o(self.fc_o(combined)))
        
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
            energy = torch.zeros(batch_size, 1).to(x.device)
            previous_x = torch.zeros(batch_size, 1).to(x.device)  # Initialize previous_x with zeros
        else:
            h, c, prev_output, energy, previous_x = states

        outputs = []
        energies = []
        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], (h, c))
            output = self.fc(torch.cat([h, x[:, t, 0].unsqueeze(1)], dim=1)) # Need to check if this is correct
            
            # Calculate and accumulate energy using the trapezoid rule
            current_x = x[:, t, 0].unsqueeze(1)
            delta_disp = current_x - previous_x
            energy = energy + (output + prev_output) / 2 * delta_disp
            energies.append(energy)
            prev_output = output
            previous_x = current_x
            outputs.append(output)

        return torch.stack(outputs, dim=1), torch.stack(energies, dim=1), (h, c, prev_output, energy, previous_x)

    

def custom_loss(y_pred, y, mask):
    loss = torch.mean((y_pred - y)**2 * mask)
    return loss

def drucker_loss(f, e, mask):
    num_samples, time_length = f.size()
    num_chops = 201
    chop_vector = torch.linspace(-1, 1, num_chops).to(f.device)
    mask_expanded = mask.unsqueeze(2).expand(-1, -1, chop_vector.size(0))
    f_expanded = f.unsqueeze(2).expand(-1, -1, chop_vector.size(0))
    e_expanded = e.unsqueeze(2).expand(-1, -1, chop_vector.size(0))
    chop_vector_expanded = chop_vector.unsqueeze(0).unsqueeze(0).expand(num_samples, time_length, -1)
    deducted_f_expanded = f_expanded - chop_vector_expanded
    # Create deducted_f_sign tensor with size [num_samples, time_length, num_chops]
    deducted_f_sign = (deducted_f_expanded > 0).int()
    
    diff_sign = torch.diff(deducted_f_sign, dim=1, prepend=deducted_f_sign[:, 0, :].unsqueeze(1))
    change_idx = ((diff_sign != 0) * mask_expanded).bool()
    selected_e = torch.where(change_idx, e_expanded, torch.full_like(e_expanded, float('nan')))

    mask_no_nan = ~torch.isnan(selected_e)

    reordered_e = selected_e.permute(2, 0, 1)
    reordered_mask = mask_no_nan.permute(2, 0, 1)

    valid_indices = torch.stack(reordered_mask.nonzero(as_tuple=True), dim=-1)

    valid_values = reordered_e[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

    diff_indices = valid_indices[1:] - valid_indices[:-1]

    differences = (valid_values[1:] - valid_values[:-1]) * (diff_indices[:, 0] == 0) * (diff_indices[:, 1] == 0)
    
    differences_neg = -differences
    negative_values_only = (torch.abs(differences_neg) + differences_neg) / 2
    
    return torch.sum(negative_values_only) / (num_samples * time_length)

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
          window_size,
          checkpoint_dir,
          title,
          checkpoint_epoch,
          existing_checkpoint=False):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1')
    
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

            loss = criterion(outputs[:, :, 0], labels, energies[:, :, 0], mask_train[:, step:step+window_size], alpha=0.5)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            states = (states[0].detach(), states[1].detach(), states[2].detach(), states[3].detach(), states[4].detach())
            
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



