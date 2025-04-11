from casadi_opti_station import ocp_station_knot, ocp_parallel
import numpy as np
from os.path import join
from os import getcwd
import os
from sys import argv
from utils import num2str
from tqdm import tqdm
from multiprocessing import Pool

# python file for generating pareto front solutions from objective function weight combinations (grid search). 

def generate_pareto_grid_parallel(knot_range=(0.001, 1000, 7),
                                  fuel_range=(0.001, 1000, 7),
                                  thrust_limit=1.0,
                                  save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions_short_path'),
                                  view_distance_str='2m'
                                  ):
    """generate solutions for varying knot and fuel cost weight values

    Args:
        knot_range (tuple, optional): knot_cost_weight limits and number of discretized values. Defaults to (0.001, 100).
        fuel_range (tuple, optional): fuel_cost_weight limits and number of discretized values. Defaults to (0.001 100).
        thrust_limit (float, optional): thrust limit for actions. Defaults to 0.5.
        save_dir (path): path to save solutions
        view_distance_str (str): string to indicate knot file correspondance
    """
    knot_weights = np.logspace(np.log10(knot_range[0]), np.log10(knot_range[1]), knot_range[2])
    fuel_weights = np.logspace(np.log10(fuel_range[0]), np.log10(fuel_range[1]), fuel_range[2])

    print('BUILDING PARALLEL TASK')
    print('Knot Weights: ', knot_weights)
    print('Fuel Weights: ', fuel_weights)

    arg_list = []
    idx=0
    for kw in knot_weights[idx:]:
        for fw in fuel_weights[idx:]:
            # parse kw and fw into a filename string
            k_str = num2str(kw)
            f_str = num2str(fw)
            filestr = 'k_' + k_str + '_f_' + f_str
            print('Current Weight Config: ', filestr)
            print('Save path: ', join(save_dir, filestr + '_X.csv'))

            args = (save_dir,
                   filestr,
                   thrust_limit,
                   kw,
                   0.0,
                   fw,
                   True,
                   view_distance_str,
                   True
                   )

            arg_list.append(args)

    with Pool(2) as p:
        r = list(tqdm(p.imap(ocp_parallel, arg_list), total=len(arg_list)))

def pareto_grid_individual_sig(
    w,
    thrust_limit=0.5,
    save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions_short_path'),
    view_distance_str='2.0m'
):
    """generate solutions for varying knot and fuel cost weight values

    Args:
        knot_range (tuple, optional): knot_cost_weight limits and number of discretized values. Defaults to (0.001, 100).
        fuel_range (tuple, optional): fuel_cost_weight limits and number of discretized values. Defaults to (0.001 100).
        thrust_limit (float, optional): thrust limit for actions. Defaults to 0.5.
        save_dir (path): path to save solutions
        view_distance_str (str): string to indicate knot file correspondance
    """

    locality = view_distance_str[-3] == 'c' # check if local

    fw = 1/(1+np.exp(-w)) # start small
    kw = 1-fw

    w_str = num2str(w)

    # k_str = num2str(kw)
    # f_str = num2str(fw)
    filestr = 'w_' + w_str
    # filestr = 'k_' + k_str + '_f_' + f_str
    print('Current Weight Config: ', filestr)
    print('Save path: ', join(save_dir, filestr + '_X.csv'))

    if not os.path.exists(save_dir): os.mkdir(save_dir)

    ocp_station_knot(save_folder=save_dir,
                        save_path=filestr,
                        thrust_limit=thrust_limit,
                        k_weight=kw,
                        # p_weight=0.1,
                        p_weight=0.0,
                        f_weight=fw,
                        closest_knot=True,
                        view_distance=view_distance_str,
                        # local=True,
                        local=locality,
                        num_knots=None
                        )


def pareto_grid_individual(kw,
                           fw,
                           thrust_limit=0.5,
                           save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions_short_path'),
                           view_distance_str='2.0m'
):
    """generate solutions for varying knot and fuel cost weight values

    Args:
        knot_range (tuple, optional): knot_cost_weight limits and number of discretized values. Defaults to (0.001, 100).
        fuel_range (tuple, optional): fuel_cost_weight limits and number of discretized values. Defaults to (0.001 100).
        thrust_limit (float, optional): thrust limit for actions. Defaults to 0.5.
        save_dir (path): path to save solutions
        view_distance_str (str): string to indicate knot file correspondance
    """

    locality = view_distance_str[-3] == 'c' # check if local

    k_str = num2str(kw)
    f_str = num2str(fw)
    filestr = 'k_' + k_str + '_f_' + f_str
    print('Current Weight Config: ', filestr)
    print('Save path: ', join(save_dir, filestr + '_X.csv'))

    if not os.path.exists(save_dir): os.mkdir(save_dir)

    ocp_station_knot(save_folder=save_dir,
                        save_path=filestr,
                        thrust_limit=thrust_limit,
                        k_weight=kw,
                        # p_weight=0.1,
                        p_weight=0.0,
                        f_weight=fw,
                        closest_knot=True,
                        view_distance=view_distance_str,
                        # local=True,
                        local=locality,
                        num_knots=None
                        )

def generate_pareto_grid(knot_range=(0.001, 1000, 7),
                         fuel_range=(0.001, 1000, 7),
                         thrust_limit=0.5,
                         save_dir=join(getcwd(), 'ocp_paths', 'pareto_front_solutions_short_path'),
                         view_distance_str='2.0m'
                         ):
    """generate solutions for varying knot and fuel cost weight values

    Args:
        knot_range (tuple, optional): knot_cost_weight limits and number of discretized values. Defaults to (0.001, 100).
        fuel_range (tuple, optional): fuel_cost_weight limits and number of discretized values. Defaults to (0.001 100).
        thrust_limit (float, optional): thrust limit for actions. Defaults to 0.5.
        save_dir (path): path to save solutions
        view_distance_str (str): string to indicate knot file correspondance
    """
    knot_weights = np.logspace(np.log10(knot_range[0]), np.log10(knot_range[1]), knot_range[2])
    fuel_weights = np.logspace(np.log10(fuel_range[0]), np.log10(fuel_range[1]), fuel_range[2])

    print('Knot Weights: ', knot_weights)
    print('Fuel Weights: ', fuel_weights)

    kwf = []
    for kw in knot_weights:
        for fw in fuel_weights:
            kwf.append((kw, fw))

    kwfleft = kwf[10:]
    print('Combinations Left: ', len(kwfleft))
    for kw, fw in kwfleft:
        # parse kw and fw into a filename string
        k_str = num2str(kw)
        f_str = num2str(fw)
        filestr = 'k_' + k_str + '_f_' + f_str
        print('Current Weight Config: ', filestr)
        print('Save path: ', join(save_dir, filestr + '_X.csv'))

        ocp_station_knot(save_folder=save_dir,
                            save_path=filestr,
                            thrust_limit=thrust_limit,
                            k_weight=kw,
                            p_weight=0.1,
                            f_weight=fw,
                            closest_knot=True,
                            view_distance=view_distance_str,
                            num_knots=5
                            )

def parse_generation_args(argv):
    """generate pareto front solutions based on argv in

    Args:
        argv (_type_): arguments submitted to the python call
    """
    if len(argv) > 1 and argv[1] == '-h': print('python pareto_grid.py thrust_limit soln_folder knot_weight_range fuel_weight_range',
                                                '\n  thrust_limit=0.5',
                                                '\n  soln_folder=pareto_front_solutions',
                                                '\n  knot_weight_range=(0.1, 100, 8)',
                                                '\n  fuel_weight_range=(0.1, 100, 8)')
    elif len(argv) > 1 and argv[1] == '-p':
        if len(argv) > 2: thrust_limit_input = float(argv[2])
        else: thrust_limit_input = 1.0
        if len(argv) > 3: soln_folder = argv[3]
        else: soln_folder = 'pf_2m_local'
        if len(argv) > 6: knot_range_input = (float(argv[4]), float(argv[5]), int(argv[6]))
        else: knot_range_input = (0.001, 1000, 7)
        if len(argv) > 9: fuel_range_input = (float(argv[7]), float(argv[8]), int(argv[9]))
        else: fuel_range_input = (0.001, 1000, 7)
        generate_pareto_grid_parallel(knot_range=knot_range_input,
                                      fuel_range=fuel_range_input,
                                      thrust_limit=thrust_limit_input,
                                      save_dir=join(getcwd(), 'ocp_paths', soln_folder),
                                      view_distance_str='2m'
                                      )
    elif len(argv) > 1 and argv[1] == '-auto':
        thrust_limit_input = 1.0
        soln_folder = 'pareto_front_2m_local'
        knot_range_input = (0.001, 1000, 13)
        fuel_range_input = (0.001, 1000, 13)
        generate_pareto_grid(knot_range=knot_range_input,
                                      fuel_range=fuel_range_input,
                                      thrust_limit=thrust_limit_input,
                                      save_dir=join(getcwd(), 'ocp_paths', soln_folder),
                                      view_distance_str='2m'
                                      )
    elif len(argv) > 1 and argv[1] == '-sig':
        thrust_limit_input = 1.0
        soln_folder = argv[3]
        w_in = float(argv[2])
        pareto_grid_individual_sig(
            w_in,
            thrust_limit=thrust_limit_input,
            save_dir=join(getcwd(), 'ocp_paths', 'pf_' + soln_folder),
            view_distance_str=soln_folder
        )
    elif len(argv) > 1 and argv[1] == '-bash':
        thrust_limit_input = 1.0
        # soln_folder = 'pareto_front_2m_local'
        # soln_folder = 'test'
        soln_folder = argv[4]
        pareto_grid_individual(kw = float(argv[2]),
                               fw = float(argv[3]),
                               thrust_limit=thrust_limit_input,
                               save_dir=join(getcwd(), 'ocp_paths', 'pf_' + soln_folder),
                               view_distance_str=argv[4])
    else:
        if len(argv) > 1: thrust_limit_input = float(argv[1])
        else: thrust_limit_input = 1.0
        if len(argv) > 2: soln_folder = argv[2]
        else: soln_folder = 'pareto_front_solutions_short_path'
        if len(argv) > 5: knot_range_input = (float(argv[3]), float(argv[4]), int(argv[5]))
        else: knot_range_input = (0.01, 100, 9)
        if len(argv) > 8: fuel_range_input = (float(argv[6]), float(argv[7]), int(argv[8]))
        else: fuel_range_input = (0.01, 100, 9)
        generate_pareto_grid(knot_range=knot_range_input, fuel_range=fuel_range_input,
                             thrust_limit=thrust_limit_input, save_dir=join(getcwd(), 'ocp_paths', soln_folder))

def single_solutions_w1():
    view_distance_strs = (
        "2m_global",
        "2m_local",
        "4m_global",
        "4m_local",
        "8m_global",
        "8m_local",
        "16m_global",
        "16m_local"
    )

    for view_distance_str in view_distance_strs:
        pareto_grid_individual_sig(
            w=1e-8,
            thrust_limit=1.0,
            save_dir=join(getcwd(), 'ocp_paths', 'single_solution_ral_comments'),
            view_distance_str=view_distance_str
        )

if __name__ == "__main__":
    single_solutions_w1()
    # parse_generation_args(argv)