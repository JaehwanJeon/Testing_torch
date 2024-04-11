#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter

data_name = 'Link_4'
data_path = os.path.join('Hysteresis/Pedram', data_name + '.csv')
data = np.genfromtxt(data_path, delimiter=',')

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Applying the low-pass filter to a signal
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



# %%
cutoff = 12
fs = 50
data_filtered = np.array([data[:, 0], butter_lowpass_filter(data[:, 1], cutoff, fs, 5)]).T
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(data[:, 0], data[:, 1], 'b', alpha=0.5)
ax.plot(data_filtered[:, 0], data_filtered[:, 1], 'r', alpha=1)




#%%
np.savetxt(os.path.join('Hysteresis/Pedram', data_name + '_filtered.csv'), data_filtered, delimiter=',')
# %%
