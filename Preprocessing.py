import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess(n_samples,
               hysteresis_data_dir,
               processed_data_dir,
               title,
               val_size=False,
               normalize_gap=False,
               test=True):
    # Total energy as input
    X = []
    y = []

    time_length = []
    num_inputs = 2  # Displacement and mask
    for i in range(n_samples):
        outputs = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}.npz'.format(i))))
        disp = outputs['disp']
        force = outputs['force']
        energy = outputs['energy']
        X.append(np.concatenate((disp[:, np.newaxis], np.ones((len(disp), 1))), axis=1))
        y.append(force)
        time_length.append(len(disp))
    time_length = np.array(time_length)
    max_time_length = np.max(time_length)
    for i in range(n_samples):
        if len(X[i]) < max_time_length:
            X[i] = np.concatenate((X[i], np.zeros((max_time_length - len(X[i]), num_inputs))), axis=0)
            y[i] = np.concatenate((y[i], np.zeros((max_time_length - len(y[i])))), axis=0)
    X, y = np.array(X), np.array(y)    # X.shape = (n_samples, max_time_length, num_inputs), y.shape = (n_samples, max_time_length)

    if normalize_gap != False:
        X_max = np.max(np.abs(X), axis=(0,1))[:num_inputs-1]
        y_max = np.max(np.abs(y))
        X[:, :, :num_inputs - 1] = X[:, :, :num_inputs - 1] / (X_max * (1 + normalize_gap))
        y = y / (y_max * (1 + normalize_gap))

    if test:
        X_train, X_test, y_train, y_test = X[:int(n_samples*0.5)], X[int(n_samples*0.5):], y[:int(n_samples*0.5)], y[int(n_samples*0.5):]

    if val_size != False:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=0)

    if test:
        np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + title + '_Processed_data.npz')), X_train=X_train, X_test=X_test, X_val=X_val, 
                y_train=y_train, y_test=y_test, y_val=y_val, normalize_gap=normalize_gap, X_max=X_max, y_max=y_max)
    else:
        np.savez(os.path.normpath(os.path.join(processed_data_dir, './' + title + '_Processed_data.npz')), X_train=X_train, X_val=X_val, 
                y_train=y_train, y_val=y_val, normalize_gap=normalize_gap, X_max=X_max, y_max=y_max)


