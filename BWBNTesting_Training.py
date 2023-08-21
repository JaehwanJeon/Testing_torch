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
          checkpoint_epoch):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:1')
    model = CustomLSTM(2, nn_size, 1).to(device)
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
            print('Step:', step, '//', X_train.shape[1]-window_size)
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

        # 일정 주기마다 손실 출력
        if (epoch + 1) % checkpoint_epoch == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
            y_pred, _ = model(X_val)
            loss = criterion(y_pred[:, :, 0], y_val, mask_val)
            print(f'Validation Loss: {loss.item()}')
            torch.save(model.state_dict(), os.path.normpath(os.path.join(checkpoint_dir, 'checkpoint_{}.pth'.format(epoch+1))))
    
    return model



# def train(nn_size,
#           num_inputs,
#           output_factor,
#           loss_weights,
#           optimizer,
#           batch_size,
#           buffer_size,
#           num_epochs,
#           model_dir,
#           processed_data_dir,
#           test_size):

#     os.makedirs(model_dir, exist_ok=True)
#     from tensorflow.python.client import device_lib
#     device_lib.list_local_devices()


#     """Load data"""
#     # Total energy as input
#     Processed_data = np.load(os.path.normpath(os.path.join(processed_data_dir, './Processed_data.npz')))
#     X_train, Xb_train, Xc_train, y_train, ye_train = Processed_data['X_train'], Processed_data['Xb_train'], Processed_data['Xc_train'], Processed_data['y_train'], Processed_data['ye_train']

#     X, Xb, Xc, y, ye = X_train, Xb_train, Xc_train, y_train, ye_train
#     train_idx, val_idx = train_test_split(range(X.shape[0]), test_size=test_size, random_state=0)
#     X_train, Xb_train, Xc_train, X_val, Xb_val, Xc_val = X[train_idx], Xb[train_idx], Xc[train_idx], X[val_idx], Xb[val_idx], Xc[val_idx]
#     y_train, y_val, ye_train, ye_val = y[train_idx], y[val_idx], ye[train_idx], ye[val_idx]

#     print(X_train.shape, Xb_train.shape, Xc_train.shape, y_train.shape)
#     print(X_val.shape, Xb_val.shape, Xc_val.shape, y_val.shape)

#     del Processed_data
    
#     Xa_train = tf.data.Dataset.from_tensor_slices(X_train[:, 1:, :])
#     Xb_train = tf.data.Dataset.from_tensor_slices(Xb_train)
#     Xc_train = tf.data.Dataset.from_tensor_slices(Xc_train)
#     Xd_train = tf.data.Dataset.from_tensor_slices(X_train[:, :, 0])
    
#     y_train = tf.data.Dataset.from_tensor_slices(y_train)
#     ye_train = tf.data.Dataset.from_tensor_slices(ye_train)


#     Xa_val = tf.data.Dataset.from_tensor_slices(X_val[:, 1:, :])
#     Xb_val = tf.data.Dataset.from_tensor_slices(Xb_val)
#     Xc_val = tf.data.Dataset.from_tensor_slices(Xc_val)
#     Xd_val = tf.data.Dataset.from_tensor_slices(X_val[:, :, 0])

#     y_val = tf.data.Dataset.from_tensor_slices(y_val)
#     ye_val = tf.data.Dataset.from_tensor_slices(ye_val)

#     train_dataset = tf.data.Dataset.zip(((Xa_train, Xb_train, Xc_train, Xd_train), (y_train, ye_train)))
#     train_dataset = train_dataset.cache("GPU")
#     train_dataset = train_dataset.batch(batch_size)
#     val_dataset = tf.data.Dataset.zip(((Xa_val, Xb_val, Xc_val, Xd_val), (y_val, ye_val)))
#     val_dataset = val_dataset.cache("GPU")
#     val_dataset = val_dataset.batch(batch_size)
#     """Generate the model"""

#     # Total energy as input
#     keras.backend.clear_session()

#     input_A = keras.layers.Input((num_inputs-1, 2))
#     input_A0 = keras.layers.Input(num_inputs)
#     norm_layer = keras.layers.Normalization(axis=-1)(input_A)

#     input_B = keras.layers.Input(2)
#     hidden_B1 = keras.layers.Dense(1)(input_B)  # (None, 1), initial condition for z

#     input_C = keras.layers.Input(1)
#     norm_layer_C = keras.layers.Normalization(axis=-1)(input_C)
#     hidden_C1 = keras.layers.Dense(1)(norm_layer_C)  # total initial energy

#     init_h = tf.repeat(hidden_B1, nn_size, axis=1)
#     init_c = tf.repeat(hidden_C1, nn_size, axis=1)

#     hidden_1 = keras.layers.LSTM(nn_size, activation='tanh', return_sequences=True)(norm_layer, initial_state=[init_h, init_c]) # z

#     output_layer_0 = keras.layers.Dense(1)(hidden_1)
#     output_layer_f = output_layer_0[:,-1]

#     ds_layer = input_A0[:, 1:]-input_A0[:, :-1]
#     pre_layer_0 = tf.concat([tf.expand_dims(input_B[:,0], -1), output_layer_0[:, :, 0]], axis=1)
#     pre_layer_1 = 1/2*(pre_layer_0[:, 1:]+pre_layer_0[:, :-1])
#     output_layer_2 = tf.math.multiply(ds_layer, pre_layer_1)*output_factor
#     output_layer_e0 = tf.reduce_sum(output_layer_2, 1, keepdims=True)
#     output_layer_e = output_layer_e0+input_C

#     DNN_model = keras.models.Model(inputs=[input_A, input_B, input_C, input_A0], outputs=[output_layer_f, output_layer_e])
#     print(DNN_model.summary())


#     """Compile the model"""
#     DNN_model.compile(loss=[['mse'], ['mse']], loss_weights=loss_weights, optimizer=optimizer)
#     model_checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.normpath(os.path.join(model_dir, './PINN_{}_{}_{}'.format(num_inputs, nn_size, 40))), save_best_only=True)


#     """Train the model"""
#     history = DNN_model.fit(train_dataset, epochs=num_epochs,
#                             validation_data=val_dataset, callbacks=[model_checkpoint_cb])