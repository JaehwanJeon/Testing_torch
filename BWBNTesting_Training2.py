"""
If successful, then add model_name='PINN_{}_{}_{}'.format(num_inputs, nn_size, 40)'
Should give random.seed
Any ways to save history?
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split


def train(nn_size,
          num_inputs,
          output_factor,
          loss_weights,
          optimizer,
          batch_size,
          buffer_size,
          num_epochs,
          model_dir,
          processed_data_dir,
          test_size):

    os.makedirs(model_dir, exist_ok=True)
    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()


    """Load data"""
    # Total energy as input
    Processed_data = np.load(os.path.normpath(os.path.join(processed_data_dir, './Processed_data.npz')))
    X_train, Xb_train, Xc_train, y_train, ye_train = Processed_data['X_train'], Processed_data['Xb_train'], Processed_data['Xc_train'], Processed_data['y_train'], Processed_data['ye_train']

    X, Xb, Xc, y, ye = X_train, Xb_train, Xc_train, y_train, ye_train
    train_idx, val_idx = train_test_split(range(X.shape[0]), test_size=test_size, random_state=0)
    X_train, Xb_train, Xc_train, X_val, Xb_val, Xc_val = X[train_idx], Xb[train_idx], Xc[train_idx], X[val_idx], Xb[val_idx], Xc[val_idx]
    y_train, y_val, ye_train, ye_val = y[train_idx], y[val_idx], ye[train_idx], ye[val_idx]

    print(X_train.shape, Xb_train.shape, Xc_train.shape, y_train.shape)
    print(X_val.shape, Xb_val.shape, Xc_val.shape, y_val.shape)

    del Processed_data
    
    Xa_train = tf.data.Dataset.from_tensor_slices(X_train[:, 1:, :])
    Xb_train = tf.data.Dataset.from_tensor_slices(Xb_train)
    Xc_train = tf.data.Dataset.from_tensor_slices(Xc_train)
    Xd_train = tf.data.Dataset.from_tensor_slices(X_train[:, :, 0])
    
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    ye_train = tf.data.Dataset.from_tensor_slices(ye_train)


    Xa_val = tf.data.Dataset.from_tensor_slices(X_val[:, 1:, :])
    Xb_val = tf.data.Dataset.from_tensor_slices(Xb_val)
    Xc_val = tf.data.Dataset.from_tensor_slices(Xc_val)
    Xd_val = tf.data.Dataset.from_tensor_slices(X_val[:, :, 0])

    y_val = tf.data.Dataset.from_tensor_slices(y_val)
    ye_val = tf.data.Dataset.from_tensor_slices(ye_val)

    train_dataset = tf.data.Dataset.zip(((Xa_train, Xb_train, Xc_train, Xd_train), (y_train, ye_train)))
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = tf.data.Dataset.zip(((Xa_val, Xb_val, Xc_val, Xd_val), (y_val, ye_val)))
    val_dataset = val_dataset.batch(batch_size)
    """Generate the model"""

    # Total energy as input
    keras.backend.clear_session()

    input_A = keras.layers.Input((num_inputs-1, 2))
    input_A0 = keras.layers.Input(num_inputs)
    norm_layer = keras.layers.Normalization(axis=-1)(input_A)

    input_B = keras.layers.Input(2)
    hidden_B1 = keras.layers.Dense(1)(input_B)  # (None, 1), initial condition for z

    input_C = keras.layers.Input(1)
    norm_layer_C = keras.layers.Normalization(axis=-1)(input_C)
    hidden_C1 = keras.layers.Dense(1)(norm_layer_C)  # total initial energy
    
    init_size = int(nn_size/2)
    init_h0 = tf.repeat(hidden_B1, init_size, axis=1)
    init_h1 = tf.zeros_like(init_h0)
    init_c0 = tf.repeat(hidden_C1, init_size, axis=1)
    init_c1 = tf.zeros_like(init_c0)
    init_h = keras.layers.Concatenate()([init_h0, init_h1])
    init_c = keras.layers.Concatenate()([init_c0, init_c1])

    hidden_1 = keras.layers.LSTM(nn_size, activation='tanh', return_sequences=True)(norm_layer, initial_state=[init_h, init_c]) # z

    output_layer_0 = keras.layers.Dense(1)(hidden_1)
    output_layer_f = output_layer_0[:,-1]

    ds_layer = input_A0[:, 1:]-input_A0[:, :-1]
    pre_layer_0 = tf.concat([tf.expand_dims(input_B[:,0], -1), output_layer_0[:, :, 0]], axis=1)
    pre_layer_1 = 1/2*(pre_layer_0[:, 1:]+pre_layer_0[:, :-1])
    output_layer_2 = tf.math.multiply(ds_layer, pre_layer_1)*output_factor
    output_layer_e0 = tf.reduce_sum(output_layer_2, 1, keepdims=True)
    output_layer_e = output_layer_e0+input_C

    DNN_model = keras.models.Model(inputs=[input_A, input_B, input_C, input_A0], outputs=[output_layer_f, output_layer_e])
    print(DNN_model.summary())


    """Compile the model"""
    DNN_model.compile(loss=[['mse'], ['mse']], loss_weights=loss_weights, optimizer=optimizer)
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.normpath(os.path.join(model_dir, './PINN_{}_{}_{}_2'.format(num_inputs, nn_size, 40))), save_best_only=True)


    """Train the model"""
    history = DNN_model.fit(train_dataset, epochs=num_epochs,
                            validation_data=val_dataset, callbacks=[model_checkpoint_cb])