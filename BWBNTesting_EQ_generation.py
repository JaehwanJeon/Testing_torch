import os
import ReadRecord
import numpy as np
import glob
import random
import Experiment_230508 as Experiment
import matplotlib.pyplot as plt


def generate_hysteresis(EQ_data_dir,
                        hysteresis_data_dir,
                        target_data,
                        change_at2_to_numpy,
                        seed,
                        gm_scale_factor,
                        n_samples,
                        draw_hysteresis,
                        mat_type,
                        mat_props,
                        EQ_list=False):
    if EQ_list == False:
        EQ_path = os.path.normpath(os.path.join(EQ_data_dir, './**/*.AT2'))
        EQ_list = glob.glob(EQ_path, recursive=True)
        random.seed(seed)
    
        EQ_list = random.sample(EQ_list, n_samples)
        random.shuffle(EQ_list)

    input_data = np.loadtxt(target_data, delimiter=' ')
    t = input_data[:, 0]
    gm = input_data[:, 1]
    dt = t[1]-t[0]

    max_value = abs(-9.8*gm*gm_scale_factor).max()
    print(max_value, 'm/s**2 is the scaled PGA for all samples')


    """Changing the EQ.AT2 file to .npy file"""
    dt_list, nPts_list = [], []
    for EQ_name in EQ_list:
        file_name, file_extension = os.path.splitext(EQ_name)
        dt, nPts = ReadRecord.ReadRecord(EQ_name, file_name+'.dat')
        dt_list.append(dt)
        nPts_list.append(nPts)
        gm = []
        with open(file_name+'.dat', 'r') as f:
            for line in f:
                if line:
                    words = line.split()
                    for word in words:
                        gm.append(float(word))
        gm = np.array(gm)*9.8
        scale_factor = max_value/abs(gm).max()
        gm = gm*scale_factor
        if change_at2_to_numpy:
            np.save(file_name+'.npy', gm)
            print('ground motions converted to numpy data')

    """Conduct analysis and save hysteresis data"""
    os.makedirs(hysteresis_data_dir, exist_ok=True)
    for i, EQ_name in enumerate(EQ_list):
        print("\r{}/{}".format(i+1, len(EQ_list)), end="")
        file_name, file_extension = os.path.splitext(EQ_name)
        gm = np.load(file_name+'.npy')
        t = np.arange(0, nPts_list[i]*dt_list[i], dt_list[i])
        outputs, beta_k = Experiment.dynamic_1DOF(mat_type, mat_props, t, gm/9.8, gm_scale=1)

        #### Calculate energy ####
        disp = outputs['rel_disp']
        force = outputs['force']
        ds = np.diff(disp, prepend=0)
        force_ = 1 / 2 * (np.concatenate(([0], force)) + np.concatenate((force, [0])))[:-1]
        energy = np.cumsum(force_ * ds)

        np.savez(os.path.normpath(os.path.join(hysteresis_data_dir, '{}.npz'.format(i))), disp=outputs['rel_disp'],
                 vel=outputs['rel_vel'], force=outputs['force'], energy=energy)
        if draw_hysteresis:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.plot(outputs['rel_disp'], outputs['force'])
            ax.set_xlabel('Displacement (m)')
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.grid()
            ax.set_title('{}'.format(i))
            fig.savefig(os.path.normpath(os.path.join(hysteresis_data_dir, './{}.png'.format(i))))
            plt.close(fig)

    np.savez(os.path.normpath(os.path.join(hysteresis_data_dir, './meta.npz')), dt_list=dt_list, nPts_list=nPts_list)
    print('Hysteresis data saved to {}'.format(hysteresis_data_dir))
