"""View the difference between the impact loading and the hysteresis loading"""

import numpy as np
import os
import torch
import torch.nn as nn
import glob
import Training
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

title = 'BWBN_h_energy_diff_disp'
hysteresis_data_dir = '/home/jaehwan/Python Project/DLCM/Testing_torch/Hysteresis'
hysteresis_data_dir_impact = os.path.normpath(os.path.join(hysteresis_data_dir, './Impact'))

hysteresis_path = os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_Processed_data.npz'))
hysteresis_path_impact = os.path.normpath(os.path.join(hysteresis_data_dir_impact, './' + title + '_Processed_data_impact.npz'))
hysteresis_disp = np.load(hysteresis_path)['X_train'][:, :, 0]
hysteresis_disp_mask = np.load(hysteresis_path)['X_train'][:, :, 2]
hysteresis_disp_impact = np.load(hysteresis_path_impact)['X_test'][:, :, 0]


for i in range(len(hysteresis_disp)):
    fig, ax = plt.subplots()
    ax.plot(hysteresis_disp[i, hysteresis_disp_mask[i].astype(bool)])
    for j in range(len(hysteresis_disp_impact)):
        ax.plot(hysteresis_disp_impact[j])
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm_Disp')
    plt.savefig(os.path.normpath(os.path.join(hysteresis_data_dir, './' + title + '_{}_disp_sample.png'.format(i))))
    plt.close()

