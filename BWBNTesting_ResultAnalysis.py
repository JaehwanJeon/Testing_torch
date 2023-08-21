import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# def result_plot(range_plot,
#                 result_plot_dir,
#                 show_plot,
#                 model_path,
#                 hysteresis_data_dir,
#                 num_inputs,
#                 output_factor):
#
#     os.makedirs(result_plot_dir, exist_ok=True)
#     DNN_model = keras.models.load_model(model_path)
#
#     for i in range(range_plot[0], range_plot[1]):
#         data_path = os.path.normpath(os.path.join(hysteresis_data_dir, './{}.npz'.format(i)))
#         save_path = os.path.normpath(os.path.join(result_plot_dir, './{}.png'.format(i)))
#         test_plot(data_path, DNN_model, num_inputs, output_factor, save_path=save_path, show_plot=show_plot)


def result_plot(range_plot,
                result_plot_dir,
                show_plot,
                model_path,
                hysteresis_data_dir,
                num_inputs,
                output_factor):

    os.makedirs(result_plot_dir, exist_ok=True)
    DNN_model = keras.models.load_model(model_path)

    # max value list are made so that the first row is reference, and the second row is testing
    max_force_list = np.zeros((2, range_plot[1]-range_plot[0]))
    max_energy_list = np.zeros((2, range_plot[1] - range_plot[0]))

    for i in range(range_plot[0], range_plot[1]):
        data_path = os.path.normpath(os.path.join(hysteresis_data_dir, './{}.npz'.format(i)))
        save_path = os.path.normpath(os.path.join(result_plot_dir, './{}.png'.format(i)))
        save_max_path = os.path.normpath(os.path.join(result_plot_dir, './{}_max.png'.format(i)))
        outputs = np.load(data_path)
        disp = outputs['disp']
        vel = outputs['vel']
        energy = outputs['energy']
        y_pred = predict_force_energy(DNN_model, disp, vel, num_inputs)
        max_force_list[0, i], max_force_list[1, i] = abs(disp).max(), abs(y_pred[0]).max()
        max_energy_list[0, i], max_energy_list[1, i] = energy.max(), abs(y_pred[1]).max()
        test_plot(data_path, DNN_model, num_inputs, output_factor, save_path=save_path, show_plot=show_plot)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].plot(max_force_list[0], max_force_list[1], 'bo')
    axs[0].grid()
    axs[0].set_xlabel('Reference (N)', fontsize=12)
    axs[0].set_ylabel('Predicted (N)', fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].set_title('Maximum force prediction', fontsize=12)

    axs[1].plot(max_energy_list[0], max_energy_list[1], 'ro')
    axs[1].grid()
    axs[1].set_xlabel('Reference (Nm)', fontsize=12)
    axs[1].set_ylabel('Predicted (Nm)', fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].set_title('Maximum energy prediction', fontsize=12)

    if save_path != False:
        fig.savefig(save_max_path)
    if show_plot == False:
        plt.close(fig)




def predict_force_energy(model, disp, vel, num_inputs):
    X_A = tf.Variable(tf.zeros((len(disp), num_inputs, 2)))
    x_pre = tf.concat((tf.zeros(num_inputs - 1), disp, tf.zeros(1)), axis=0)
    xdot_pre = tf.concat((tf.zeros(num_inputs - 1), vel, tf.zeros(1)), axis=0)
    for i in range(num_inputs):
        X_A[:, i, 0].assign(x_pre[i:-num_inputs + i])
        X_A[:, i, 1].assign(xdot_pre[i:-num_inputs + i])

    f_hat = []
    e_hat = []
    X_B0 = tf.Variable(tf.zeros((num_inputs - 1, 2)))
    X_C0 = tf.Variable(tf.zeros((num_inputs - 1, 1)))
    for ii in range(len(disp) // (num_inputs - 1)):
        print(ii, '/', len(disp) // (num_inputs - 1))
        X_B0[:, 1].assign(X_A[ii * (num_inputs - 1):(ii + 1) * (num_inputs - 1), 0, 0])
        XA = X_A[ii * (num_inputs - 1):(ii + 1) * (num_inputs - 1)]
        XB = X_B0
        XC = X_C0
        prediction = model.call((XA[:, 1:, :], XB, XC, XA[:, :, 0]))
        f_hat.append(prediction[0][:, 0])
        e_hat.append(prediction[1][:, 0])
        X_B0[:, 0].assign(f_hat[ii])
        X_C0[:, 0].assign(e_hat[ii])
    if len(disp) % (num_inputs - 1) != 0:
        last = len(disp) % (num_inputs - 1)
        X_B0 = tf.Variable(tf.zeros((last, 2)))
        X_C0 = tf.Variable(tf.zeros((last, 1)))
        X_B0[:, 1].assign(X_A[(ii + 1) * (num_inputs - 1):(ii + 1) * (num_inputs - 1) + last, 0, 0])
        X_B0[:, 0].assign(f_hat[-1][:last])
        X_C0[:, 0].assign(e_hat[-1][:last])
        XA = X_A[(ii + 1) * (num_inputs - 1):(ii + 1) * (num_inputs - 1) + last]
        XB = X_B0
        XC = X_C0
        prediction = model.call((XA[:, 1:, :], XB, XC, XA[:, :, 0]))
        f_hat.append(prediction[0][:, 0])
        e_hat.append(prediction[1][:, 0])
        predict_f = tf.concat((tf.reshape(tf.stack(f_hat[:-1]), [-1]), f_hat[-1]), axis=0)
        predict_e = tf.concat((tf.reshape(tf.stack(e_hat[:-1]), [-1]), e_hat[-1]), axis=0)
    else:
        predict_f = tf.reshape(tf.stack(f_hat), [-1])
        predict_e = tf.reshape(tf.stack(e_hat), [-1])
    return predict_f.numpy(), predict_e.numpy()


def draw_test_plot(outputs, y_pred, output_factor, save_path=False, show_plot=True):

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    disp = outputs['disp']
    y_test = outputs['force']/output_factor
    """Later, please change this to
    y_test = outputs['force']"""

    axs[0].plot(disp, y_test)
    axs[0].plot(disp, y_pred[0])
    axs[0].grid()
    axs[0].set_xlabel('Displacement (m)', fontsize=12)
    axs[0].set_ylabel('Force (N)', fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].legend(['Reference', 'Predicted'], fontsize=12)

    ds = np.diff(disp, prepend=0)
    force = y_test * output_factor
    force_ = 1 / 2 * (np.concatenate(([0], force)) + np.concatenate((force, [0])))[:-1]
    e_test = np.cumsum(force_ * ds)
    """Later, please change this into e_test = outputs['energy']"""

    axs[1].plot(e_test)
    axs[1].plot(y_pred[1])
    axs[1].grid()
    axs[1].set_ylabel('Total energy (Nm)', fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].legend(['Reference', 'Predicted'], fontsize=12)

    if save_path != False:
        fig.savefig(save_path)
    if show_plot == False:
        plt.close(fig)

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.plot(disp, y_test - y_pred[0])
    # ax.grid()
    # ax.set_xlabel('Displacement (m)', fontsize=12)
    # ax.set_ylabel('Force (N)', fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.legend(['Prediction error'], fontsize=12)
    # if save_dir != False:
    #     fig.savefig(save_dir + '/{}_prediction_error.png'.format(idx))
    # if show_plot == False:
    #     plt.close()


def test_plot(data_path, DNN_model, num_inputs, output_factor, save_path=False, show_plot=True):
    outputs = np.load(data_path)
    disp = outputs['disp']
    vel = outputs['vel']
    y_pred = predict_force_energy(DNN_model, disp, vel, num_inputs)

    draw_test_plot(outputs, y_pred, output_factor, save_path, show_plot)


