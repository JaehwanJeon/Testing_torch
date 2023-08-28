"""
If successful, then add model_name='PINN_{}_{}_{}'.format(num_inputs, nn_size, 40)'
Should give random.seed
Any ways to save history?
"""

import os
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
        # Forget gate
        self.fc_f = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        # Cell state
        self.fc_c = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        # Output gate
        self.fc_o = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)

    def forward(self, x, states):
        h, c = states
        
        combined = torch.cat([x, h], 1)  # concatenate along the feature dimension

        i = torch.sigmoid(self.fc_i(combined))
        f = torch.sigmoid(self.fc_f(combined))
        g = torch.tanh(self.fc_c(combined))
        o = torch.sigmoid(self.fc_o(combined))
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next) + h    # Adding residual connection

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
            h, c = torch.zeros(batch_size, self.hidden_dim).to(x.device), torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h, c = states

        outputs = []
        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], (h, c))
            output = self.fc(torch.cat([h, x[:, t, 0].unsqueeze(1)], dim=1)) # Need to check if this is correct
            outputs.append(output)
        return torch.stack(outputs, dim=1), (h, c)
    

def custom_loss(y_pred, y, mask):
    loss = torch.mean((y_pred - y)**2 * mask)
    return loss

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

    criterion = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_train, y_train, mask_train = torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device), torch.from_numpy(mask_train).float().to(device)
    X_val, y_val, mask_val = torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).float().to(device), torch.from_numpy(mask_val).float().to(device)


    # Training loop
    for epoch in range(num_epochs):
        states = None
        losses = []
        print('Epoch:', epoch+1)
        for step in range(0, X_train.shape[1]-window_size, window_size):
            print('Step:', step, '//', X_train.shape[1]-window_size, end='\r')
            inputs = X_train[:, step:step+window_size]
            labels = y_train[:, step:step+window_size]

            outputs, states = model(inputs, states)
            loss = criterion(outputs[:, :, 0], labels, mask_train[:, step:step+window_size])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            states = (states[0].detach(), states[1].detach())
            
        avg_loss = sum(losses) / len(losses)
        
        val_loss = []
        # 일정 주기마다 손실 출력
        if (epoch + 1) % checkpoint_epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
            y_pred, _ = model(X_val)
            loss = criterion(y_pred[:, :, 0], y_val, mask_val)
            val_loss.append(loss.item())
            print(f'Validation Loss: {loss.item()}')
            torch.save(model.state_dict(), os.path.normpath(os.path.join(checkpoint_dir, title + '_checkpoint_LARGENN_{}.pth'.format(epoch+1))))
        
        np.savez(os.path.normpath(os.path.join(checkpoint_dir, title + '_losses_LARGENN.npz')), val_loss=val_loss)
    return model



