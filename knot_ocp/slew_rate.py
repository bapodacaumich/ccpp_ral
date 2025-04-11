import numpy as np
import os
from utils import angle_between, num2str
import matplotlib.pyplot as plt

# compute slew rate over time (degrees per second of rotation)

def compute_slew_rate_traj(X):
    """compute the slew rate for trajectory X

    Args:
        X (np.ndarray): trajectory data (n x 7) - (x, y, z, v0, v1, v2, t) where (v0, v1, v2) is a view direction unit vector
    Returns:
        slew_rates (list): list of slew rates (degrees per second) for each time step
    """

    slew_rates = []
    for i in range(1, X.shape[0]):
        # find time difference between time steps
        dt = X[i, -1] - X[i-1, -1]

        # find angle between view directions in degrees
        dtheta = angle_between(X[i-1, 3:6], X[i, 3:6]) * 180 / np.pi 

        # find slew rate in degrees per second
        slew_rates.append(dtheta / dt)
    return slew_rates, X[:-1, -1]

def compute_slew_rates_file(file):
    """compute the slew rates for a trajectory file

    Args:
        file (str): path to trajectory file
    Returns:
        slew_rates (list): list of slew rates (degrees per second) for each time step
    """

    X = np.loadtxt(file, delimiter=',')
    return compute_slew_rate_traj(X)

def compute_slew_rates_cw_opt(condition):
    """plot slew rates for all trajectories given a condition. paths are assumed "cw_opt_packaged_" + condition

    Args:
        condition (string): cw path max time condition
    """

    files = [
        '2m_global',
        '2m_local',
        '4m_global',
        '4m_local',
        '8m_global',
        '8m_local',
        '16m_global',
        '16m_local'
    ]

    dir = os.path.join(os.getcwd(), 'cw_opt_packaged_' + condition)

    savedir = os.path.join(os.getcwd(), 'slew_rate_plots', 'cw_opt_' + condition)

    if not os.path.exists(savedir): os.makedirs(savedir)

    print('IVT t=' + condition + ' : Slew Rates (deg/s)')
    print('Condition, Mean Slew Rate (deg/s), Max Slew Rate (deg/s)')
    for qfile in files:
        for file in os.listdir(dir):
            if qfile in file:
                filepath = os.path.join(dir, file)

                sr, t = compute_slew_rates_file(filepath)
                print(qfile, ',', np.mean(sr), ',', np.max(sr))

                fig, ax = plt.subplots(figsize=(10, 3))

                ax.plot(t, sr)
                ax.set_title('Slew Rates for IVT: ' + os.path.basename(os.path.normpath(file)))
                ax.set_ylabel('Slew Rate (degrees per second)')
                ax.set_xlabel('Time (s)')

                ax.grid(True)
                # ax.set_ylim([0, 20])

                plt.tight_layout()
                fig.savefig(os.path.join(savedir, 'slew_rates_' + os.path.basename(os.path.normpath(file))[:-4] + '.png'), dpi=300)

                plt.close(fig)

def plot_slew_rates_condition(condition, save_folder):
    """plot slew rates for all trajectories given a condition

    Args:
        condition (string): folder name for condition
        save_folder (string): folder to save plots
    """
    list_conditions = [
        '2m_global',
        '2m_local',
        '4m_global',
        '4m_local',
        '8m_global',
        '8m_local',
        '16m_global',
        '16m_local'
    ]
    # get load directories
    dir = os.path.join(os.getcwd(), 'packaged_paths', condition)
    sodir = os.path.join(os.getcwd(), 'packaged_paths_so', condition)

    # generate save folder
    if not os.path.exists(os.path.join(save_folder, condition)):
        os.makedirs(os.path.join(save_folder, condition))

    # iterate through files and plot slew rates
    print('Slew Rates (deg/s) for ' + condition)
    print('Condition, Mean Slew Rate (deg/s) KO, Mean Slew Rate (deg/s) SO, Max Slew Rate (deg/s) KO, Max Slew Rate (deg/s) SO')
    for qfile in list_conditions:
        for file in os.listdir(dir):
            if qfile in file:
                # load path files
                filepath = os.path.join(dir, file)
                sofilepath = os.path.join(sodir, file)

                # compute slew rates
                sr, t = compute_slew_rates_file(filepath)
                sosr, t = compute_slew_rates_file(sofilepath)
                print(qfile, ',', np.mean(sr), ',', np.mean(sosr), ',', np.max(sr), ',', np.max(sosr))

                # plot slew rates
                fig, ax = plt.subplots(figsize=(10, 3))

                ax.plot(t, sr, label='KO')
                ax.plot(t, sosr, label='SO')
                ax.set_title('Slew Rates for ' + os.path.basename(os.path.normpath(file)))
                ax.set_ylabel('Slew Rate (degrees per second)')
                ax.set_xlabel('Time (s)')
                ax.legend()

                ax.grid(True)
                # ax.set_ylim([0, 20])

                # save plot
                plt.tight_layout()
                fig.savefig(os.path.join(save_folder, condition, 'slew_rates_' + os.path.basename(os.path.normpath(file))[:-4] + '.png'), dpi=300)

                # close plot
                plt.close(fig)

def plot_all_pf_solns():
    conditions = [
        'pf_2m_local',
        'pf_2m_global',
        'pf_4m_local',
        'pf_4m_global',
        'pf_8m_local',
        'pf_8m_global',
        'pf_16m_local',
        'pf_16m_global'
    ]
    for condition in conditions:
        plot_slew_rates_condition(condition, os.path.join(os.getcwd(), 'slew_rate_plots'))

def plot_all_cw_opt_solns():
    condition = [
        '10',
        '50',
        'var'
    ]
    for c in condition:
        compute_slew_rates_cw_opt(c)

def compute_slew_rates_pf(method):
    path_dir = os.path.join(os.getcwd(), 'packaged_paths_' + method, 'pf_final')
    conditions = [
        '2m_global',
        '2m_local',
        '4m_global',
        '4m_local',
        '8m_global',
        '8m_local',
        '16m_global',
        '16m_local'
    ]

    print('Slew Rates for ' + method, ' in directory: ', path_dir)
    print('condition, mean_slew_rate, max_slew_rate')
    for c in conditions:
        for file in os.listdir(path_dir):
            if c in file:
                filepath = os.path.join(path_dir, file)
                sr, t = compute_slew_rates_file(filepath)
                print(c, ',', np.mean(sr), ',', np.max(sr))

def pareto_slew_rate(method, condition):

    # translate pareto front directory full name method to path directory
    pfmethods = {
        'face_oriented': 'fo',
        'knot_oriented': 'ko',
        'knot_oriented_slerp': 'ko_slerp',
        'moving_average' : 'ma',
        'moving_average_slerp' : 'ma_slerp',
        'station_oriented': 'so'
    }

    save_file = os.path.join(os.getcwd(), 'pareto_front', method, condition, 'srcost.csv')
    path_dir = os.path.join(os.getcwd(), 'packaged_paths_' + pfmethods[method], condition)
    weight_file = os.path.join(os.getcwd(), 'pareto_front', method, condition, 'wcomb.csv')
    wcomb = np.loadtxt(weight_file, delimiter=' ')

    all_sr = []
    for i in range(wcomb.shape[0]):
        kw = wcomb[i,0]
        fw = wcomb[i,1]
        file = 'k_' + num2str(kw) + '_f_' + num2str(fw) + '.csv'
        filepath = os.path.join(path_dir, file)
        sr, t = compute_slew_rates_file(filepath)
        all_sr.append([np.mean(sr), np.max(sr)])

    np.savetxt(save_file, np.array(all_sr), delimiter=',')

def method_pareto_slew_rates(method):
    conditions = [
        'pf_2m_global',
        'pf_2m_local',
        'pf_4m_global',
        'pf_4m_local',
        'pf_8m_global',
        'pf_8m_local',
        'pf_16m_global',
        'pf_16m_local'
    ]
    for c in conditions:
        pareto_slew_rate(method, c)

if __name__ == "__main__":
    # plot_slew_rates_condition('pf_final', os.path.join(os.getcwd(), 'slew_rate_plots'))
    # compute_slew_rates_cw_opt('10')
    # compute_slew_rates_cw_opt('50')
    # compute_slew_rates_cw_opt('var')
    # compute_slew_rates_pf('ko')
    # compute_slew_rates_pf('ko_slerp')
    # compute_slew_rates_pf('ma')
    # compute_slew_rates_pf('ma_slerp')
    # compute_slew_rates_pf('so')
    # plot_all_cw_opt_solns()
    method_pareto_slew_rates('face_oriented')