import numpy as np
import os


def preprocess(num_inputs,
               output_factor,
               n_samples,
               hysteresis_data_dir,
               processed_data_dir):
    # Total energy as input
    X_pre = []
    Xdot_pre = []
    Xb_pre = []
    Xc_pre = []
    y_pre = []
    ye_pre = []
    for i in range(n_samples):
        outputs = np.load(os.path.normpath(os.path.join(hysteresis_data_dir, './{}.npz'.format(i))))
        disp = outputs['disp']
        vel = outputs['vel']
        force = outputs['force']
        energy = outputs['energy']
        x = np.zeros((len(disp), num_inputs))
        xdot = np.zeros((len(vel), num_inputs))

        x_pre = np.concatenate((np.zeros(num_inputs - 1), disp, np.zeros(1)))
        xdot_pre = np.concatenate((np.zeros(num_inputs - 1), vel, np.zeros(1)))
        xb_pre = np.concatenate((np.zeros(num_inputs - 1), force, np.zeros(1)))
        e_pre = np.concatenate((np.zeros(num_inputs - 1), energy, np.zeros(1)))

        for ii in range(num_inputs):
            x[:, ii] = x_pre[ii:-num_inputs + ii]
            xdot[:, ii] = xdot_pre[ii:-num_inputs + ii]
        X_pre.append(x)
        Xb_pre.append(xb_pre[:-num_inputs])
        Xc_pre.append(e_pre[:-num_inputs])
        Xdot_pre.append(xdot)
        y_pre.append(force)
        ye_pre.append(energy)

    X_train_pre, Xdot_train_pre, Xb_train_pre, Xc_train_pre = X_pre[:int(n_samples / 2)], Xdot_pre[:int(
        n_samples / 2)], Xb_pre[:int(n_samples / 2)], Xc_pre[:int(n_samples / 2)]
    X_test_pre, Xdot_test_pre, Xb_test_pre, Xc_test_pre = X_pre[int(n_samples / 2):], Xdot_pre[
                                                                                         int(n_samples / 2):], Xb_pre[
                                                                                                              int(n_samples / 2):], Xc_pre[
                                                                                                                                   int(n_samples / 2):]
    y_train_pre = y_pre[:int(n_samples / 2)]
    y_test_pre = y_pre[int(n_samples / 2):]
    ye_train_pre = ye_pre[:int(n_samples/2)]
    ye_test_pre = ye_pre[int(n_samples/2):]

    len_train_list = [0]
    len_test_list = [0]
    for i in range(len(X_train_pre)):
        len_train_list.append(len_train_list[-1] + len(X_train_pre[i]))
    for i in range(len(X_test_pre)):
        len_test_list.append(len_test_list[-1] + len(X_test_pre[i]))

    # 2nd method
    X_train, Xb_train, Xc_train = np.zeros((len_train_list[-1], num_inputs, 2)), np.zeros((len_train_list[-1], 2)), np.zeros((len_train_list[-1], 1))
    X_test, Xb_test, Xc_test = np.zeros((len_test_list[-1], num_inputs, 2)), np.zeros((len_test_list[-1], 2)), np.zeros((len_test_list[-1], 1))
    y_train, y_test = np.zeros((len_train_list[-1])), np.zeros((len_test_list[-1]))
    ye_train, ye_test = np.zeros((len_train_list[-1])), np.zeros((len_test_list[-1]))

    dum = 0
    for i in range(len(X_train_pre)):
        X_train[dum:dum + len(X_train_pre[i]), :, 0] = X_train_pre[i]
        X_train[dum:dum + len(X_train_pre[i]), :, 1] = Xdot_train_pre[i]
        Xb_train[dum:dum + len(X_train_pre[i]), 0] = Xb_train_pre[i]
        Xb_train[dum:dum + len(X_train_pre[i]), 1] = X_train_pre[i][:, 0]
        Xc_train[dum:dum + len(X_train_pre[i]), 0] = Xc_train_pre[i]
        y_train[dum:dum + len(X_train_pre[i])] = y_train_pre[i]
        ye_train[dum:dum + len(X_train_pre[i])] = ye_train_pre[i]
        dum += len(X_train_pre[i])

    dum = 0
    for i in range(len(X_test_pre)):
        X_test[dum:dum + len(X_test_pre[i]), :, 0] = X_test_pre[i]
        X_test[dum:dum + len(X_test_pre[i]), :, 1] = Xdot_test_pre[i]
        Xb_test[dum:dum + len(X_test_pre[i]), 0] = Xb_test_pre[i]
        Xb_test[dum:dum + len(X_test_pre[i]), 1] = X_test_pre[i][:, 0]
        Xc_test[dum:dum + len(X_test_pre[i]), 0] = Xc_test_pre[i]
        y_test[dum:dum + len(X_test_pre[i])] = y_test_pre[i]
        ye_test[dum:dum + len(X_test_pre[i])] = ye_test_pre[i]
        dum += len(X_test_pre[i])

    # Total energy as input --> should be changed into a more simple form
    np.savez(os.path.normpath(os.path.join(processed_data_dir, './Processed_data.npz')), X_train=X_train,
             Xb_train=Xb_train, Xc_train=Xc_train, y_train=y_train, ye_train=ye_train,
             X_test=X_test, Xb_test=Xb_test, Xc_test=Xc_test, y_test=y_test, ye_test=ye_test,
             len_train_list=np.array(len_train_list), len_test_list=len_test_list)
