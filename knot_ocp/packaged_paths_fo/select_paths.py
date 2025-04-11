# This script selects the final paths from the pareto front grid data and saves them in the pf_final directory

import os
import numpy as np

def num2str(num):
    """ parse num into hundredth palce string 123.45678900000 --> 123_45. works for numbers under 1000 and equal to or above 0.01

    Args:
        num (float): float to parse into string
    """
    if num < 0: return 'n' + num2str(-num)
    string = str(int(num)) + '_'
    num = np.round(num % 1, decimals=3)
    if num == 0: string += '0'
    else: string += str(num)[2:5]
    return string

def select_paths(save_dir='pf_final'):
    weights = [
        [1,	1000],
        [0.001,	0.1],
        [0.1,	100],
        [0.001,	1]
    ]

    # weights = [
    #     [10,    1000],
    #     [0.1,	10],
    #     [1,	    1000],
    #     [0.001,	0.1],
    #     [0.1,	1000],
    #     [0.001,	1],
    #     [0.001,	10],
    #     [0.001,	10]
    # ]

    conditions = [
        '4m_global',
        '4m_local',
        '8m_global',
        '8m_local'
    ]

    os.makedirs(save_dir, exist_ok=True)

    for w, c in zip(weights, conditions):
        # get filenames
        load_filename = os.path.join(os.getcwd(), 'pf_' + c, 'k_' + num2str(w[0]) + '_f_' + num2str(w[1]) + '.csv')
        save_filename = os.path.join(os.getcwd(), save_dir, c + '.csv')

        # load and save data
        path_data = np.loadtxt(load_filename, delimiter=',')
        np.savetxt(save_filename, path_data, delimiter=',')

if __name__ == '__main__':
    select_paths('study_paths')