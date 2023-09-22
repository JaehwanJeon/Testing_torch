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

def drucker_loss(f, e, bool_mask):
    num_samples = f.size()[0]
    loss = torch.zeros(num_samples).to(f.device)
    zero_tensor = torch.tensor(0.0).to(f.device)
    for sample_idx in range(num_samples):
        f_sample = f[sample_idx, bool_mask[sample_idx]]
        e_sample = e[sample_idx, bool_mask[sample_idx]]
        time_length = f_sample.size()[0]

        e_fl_values = torch.zeros(time_length).to(f.device)
        e_bl_values = torch.zeros(time_length).to(f.device)

        for i in range(time_length):
            # Forward loop
            for j in range(i + 1, time_length - 1):
                if f_sample[j] == f_sample[i] == f_sample[j + 1]:
                    e_fl_values[i] = e_sample[j] - e_sample[i]
                    break
                elif (f_sample[j] < f_sample[i] <= f_sample[j + 1]) or (f_sample[j] > f_sample[i] >= f_sample[j + 1]):
                    e_fl_values[i] = lin_interp(f_sample[i], (f_sample[j], e_sample[j]), (f_sample[j + 1], e_sample[j + 1])) - e_sample[i]
                    break

            # Backward loop
            for k in range(i - 1, 0, -1):
                if f_sample[k] == f_sample[i] == f_sample[k + 1]:
                    e_bl_values[i] = e_sample[i] - e_sample[k]
                    break
                elif (f_sample[k] < f_sample[i] <= f_sample[k + 1]) or (f_sample[k] > f_sample[i] >= f_sample[k + 1]):
                    e_bl_values[i] = e_sample[i] - lin_interp(f_sample[i], (f_sample[k], e_sample[k]), (f_sample[k + 1], e_sample[k + 1]))
                    break
            loss[sample_idx] = loss[sample_idx] + torch.max(zero_tensor, -e_fl_values[i]) + torch.max(zero_tensor, -e_bl_values[i])
    # f, e = f[bool_mask], e[bool_mask]
    # num_samples, time_length = f.size()
    # loss = torch.zeros(num_samples).to(f.device)

    # for sample_idx in range(num_samples):
    #     e_fl_values = torch.zeros(time_length).to(f.device)
    #     e_bl_values = torch.zeros(time_length).to(f.device)

    #     for i in range(time_length):
    #         # Forward loop
    #         for j in range(i + 1, time_length - 1):
    #             if f[sample_idx, j] == f[sample_idx, i] == f[sample_idx, j + 1]:
    #                 e_fl_values[i] = e[sample_idx, j] - e[sample_idx, i]
    #                 break
    #             elif (f[sample_idx, j] < f[sample_idx, i] <= f[sample_idx, j + 1]) or (f[sample_idx, j] > f[sample_idx, i] >= f[sample_idx, j + 1]):
    #                 e_fl_values[i] = lin_interp(f[sample_idx, i], f[sample_idx, j], e[sample_idx, j], f[sample_idx, j + 1], e[sample_idx, j + 1]) - e[sample_idx, i]
    #                 break

    #         # Backward loop
    #         for k in range(i - 1, 0, -1):
    #             if f[sample_idx, k] == f[sample_idx, i] == f[sample_idx, k + 1]:
    #                 e_bl_values[i] = e[sample_idx, i] - e[sample_idx, k]
    #                 break
    #             elif (f[sample_idx, k] < f[sample_idx, i] <= f[sample_idx, k + 1]) or (f[sample_idx, k] > f[sample_idx, i] >= f[sample_idx, k + 1]):
    #                 e_bl_values[i] = e[sample_idx, i] - lin_interp(f[sample_idx, i], f[sample_idx, k], e[sample_idx, k], f[sample_idx, k + 1], e[sample_idx, k + 1])
    #                 break

    #         loss[sample_idx] += torch.max(torch.tensor(0.0).to(f.device), -e_fl_values[i]) + torch.max(torch.tensor(0.0).to(f.device), -e_bl_values[i])

    return torch.mean(loss)

def combined_loss(y_pred, y, energies, mask, alpha=0.5):
    mse_loss = torch.mean((y_pred - y)**2 * mask)
    bool_mask = mask.bool()
    phys_loss = drucker_loss(y_pred, energies, bool_mask)  # Assuming y_pred and y are 2D tensors [batch_size x seq_length]
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

    criterion = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_train, y_train, mask_train = torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device), torch.from_numpy(mask_train).float().to(device)
    X_val, y_val, mask_val = torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).float().to(device), torch.from_numpy(mask_val).float().to(device)

    val_loss = []

    # Training loop
    for epoch in range(num_epochs):
        states = None
        losses = []
        print('Epoch:', epoch+1)
        for step in range(0, X_train.shape[1]-window_size, window_size):
            print('Step:', step, '//', X_train.shape[1]-window_size, end='\r')
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
        
        # 일정 주기마다 손실 출력
        if (epoch + 1) % checkpoint_epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
            outputs, energies,  _ = model(X_val)
            loss = criterion(outputs[:, :, 0], y_val, energies, mask_val, alpha=0.5)
            val_loss.append(loss.item())
            print(f'Validation Loss: {loss.item()}')
            torch.save(model.state_dict(), os.path.normpath(os.path.join(checkpoint_dir, title + '_checkpoint_{}.pth'.format(epoch+1))))
        
    np.savez(os.path.normpath(os.path.join(checkpoint_dir, title + '_losses.npz')), train_loss=avg_loss, val_loss=val_loss)
    return model



def lin_interp(x, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
