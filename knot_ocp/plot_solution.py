import os 
from os import getcwd, listdir
from os.path import join
from stl import mesh
from mpl_toolkits import mplot3d
from utils import filter_path_na, set_aspect_equal_3d, process_data, load_knots, draw_camera
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import numpy as np
from sys import argv

def plot_station(scale=4):
    # load station offset
    translation = np.loadtxt('translate_station.txt', delimiter=',').reshape(1,1,3)
    translation -= np.array([2.92199515, 5.14701097, 2.63653781])

    # Create a new plot
    figure = plt.figure(figsize=(12,10))
    axes = figure.add_subplot(projection='3d')
    # axes = mplot3d.Axes3D(figure)

    for i in range(15):
        meshfile = join(getcwd(), 'model', 'convex_detailed_station', str(i) + '.stl')

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(meshfile)
        vectors = (your_mesh.vectors + translation) * scale
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(vectors, alpha=0.2))
        wf = vectors.reshape(-1, 3)
        axes.plot(wf[:,0], wf[:,1], wf[:,2], 'k', lw=0.07)

    return figure, axes

def plot_knotpoints(axes, distance, local, start=None):
    """_summary_

    Args:
        axes (_type_): _description_
        distance (_type_): _description_
        local (_type_): _description_
        start (_type_, optional): switch knot start point to user inputted 'start'. Defaults to None.

    Returns:
        _type_: _description_
    """
    knotfile = join(getcwd(), 'ccp_paths', distance + ('_local' if local else '') + '.csv')
    # for file in listdir(join(getcwd(), 'ccp_paths')):
    #     if (distance == file[:4]):
    #         if not ((file[3] == 'l') ^ local):
    #             knotfile=join(getcwd(), 'ccp_paths', file)
    # knotfile=join(getcwd(), 'ccp_paths', '1.5m43.662200005359864.csv')
    # load knot points
    path = np.loadtxt(knotfile, delimiter=',') # (N, 6)
    knots = filter_path_na(path) # get rid of configurations with nans

    if start is not None:
        knots[0,:3] = start

    # print('\nFirst two Knot points: ')
    # print(knots[:2,:3])
    # knots = knots[:2,:]
    # plot knot points
    axes.plot(knots[:,0], knots[:,1], knots[:,2],'k--')
    w = np.arange(0,1,1/knots.shape[0])
    axes.scatter(knots[:,0], knots[:,1], knots[:,2], c=w,cmap='inferno', alpha=1, s=25)
    # axes.scatter(knots[:,0], knots[:,1], knots[:,2])
    # axes.scatter(knots[0,0], knots[0,1], knots[0,2], color='tab:red', label='Start', marker='x')
    # axes.scatter(knots[-1,0], knots[-1,1], knots[-1,2], color='tab:red', label='End', marker='x')
    return axes

def plot_path(axes, X, U, s=5, qs=1, view_direction=False, vs=2):
    """plot path on 3d axes

    Args:
        axes (matplotlib.axes): matplotlib 3d axes
        X (np.array): path points (N, 6)
        U (np.array): action steps (N, 3)
        s (int, optional): action quiver arrow length. Defaults to 5.
        qs (int, optional): quiver sparsity (steps to skip between arrows). Defaults to 1.

    Returns:
        axes (matplotlib.axes): matplotlib axes object
    """
    # plot path
    axes.plot(X[:,0], X[:,1], X[:,2], 'k-')
    axes.quiver(X[:-1:qs,0],X[:-1:qs,1],X[:-1:qs,2],
                s*U[::qs,0],s*U[::qs,1],s*U[::qs,2],
                color='tab:red',
                label='Thrust')

    # plot view direction
    if view_direction:
        # view_scale = 0.5
        # axes.quiver(X[::vs,0],X[::vs,1],X[::vs,2],
        #             view_scale*X[::vs,3],view_scale*X[::vs,4],view_scale*X[::vs,5],
        #             color='k',
        #             label='Thrust')

        for i in range(0, X.shape[0], vs):
            ## need to import knot points for this:
            axes = draw_camera(axes, X[i,:3], X[i,3:6], 0.5)

    # axis labels
    axes.set_xlabel('X Axis')
    axes.set_ylabel('Y Axis')
    axes.set_zlabel('Z Axis')
    return axes

def plot_two_solutions(soln_dir1, soln_dir2, distance='1.5m', local=False):
    figure, axes = plot_station()
    axes = plot_knotpoints(axes, distance, local)

    # account for local paths
    if local:
        soln_dir1 += '_local'
        soln_dir2 += '_local'

    X1 = np.loadtxt(soln_dir1 + '_X.csv', delimiter=' ')
    U1 = np.loadtxt(soln_dir1 + '_U.csv', delimiter=' ')
    t1 = np.loadtxt(soln_dir1 + '_t.csv', delimiter=' ')
    dt1 = np.diff(t1)

    X2 = np.loadtxt(soln_dir2 + '_X.csv', delimiter=' ')
    U2 = np.loadtxt(soln_dir2 + '_U.csv', delimiter=' ')
    t2 = np.loadtxt(soln_dir2 + '_t.csv', delimiter=' ')
    dt2 = np.diff(t2)

    # axes = plot_path(axes, X1, U1*dt1.reshape((-1,1)), view_direction=True)
    # axes = plot_path(axes, X2, U2*dt2.reshape((-1,1)), view_direction=True)
    axes = set_aspect_equal_3d(axes)

    plt.show()

def plot_solution(soln_dir='thrust_test_k_1_p_1_f_1', soln_file=None, thrust_limit=0.2, local=False, distance='1.5m', processing=True):
    figure, axes = plot_station()
    axes = plot_knotpoints(axes, distance, local)

    # # load path
    if soln_file is None:
        thrust_str = str(thrust_limit)[0] + '_' + str((thrust_limit%1)*10)[0] + str(((thrust_limit*10)%1)*10)[0]
        X = np.loadtxt(join(getcwd(), 'ocp_paths', soln_dir, '1.5m_X_' + thrust_str + '.csv'), delimiter=' ')
        U = np.loadtxt(join(getcwd(), 'ocp_paths', soln_dir, '1.5m_U_' + thrust_str + '.csv'), delimiter=' ')
        t = np.loadtxt(join(getcwd(), 'ocp_paths', soln_dir, '1.5m_t_' + thrust_str + '.csv'), delimiter=' ')
        soln_file = '1.5m_' + thrust_str
    else:
        X = np.loadtxt(soln_file + '_X.csv', delimiter=' ')
        U = np.loadtxt(soln_file + '_U.csv', delimiter=' ')
        t = np.loadtxt(soln_file + '_t.csv', delimiter=' ')

    # process to find normals
    knots = load_knots(distance, local)
    X = process_data(knots, X, t)

    axes = plot_path(axes, X, U, view_direction=True, qs=2, vs=10)
    axes.set_axis_off()
    axes = set_aspect_equal_3d(axes)
    savefile = os.path.basename(os.path.normpath(soln_file))
    save_dpi = 600
    plt.savefig(os.path.join(getcwd(), 'path_figures', savefile + '.png'), dpi=save_dpi)
    # print('saving: ', savepath)
    # figure.savefig(savepath, dpi=1000)

    # view_num = 0
    # axes.view_init(elev=30, azim=30)
    # plt.savefig(os.path.join(getcwd(), 'path_figures', savefile + str(view_num) + '.png'), dpi=save_dpi)

    # # view from underneath
    # axes.view_init(elev=-30, azim=30)
    # view_num += 1
    # plt.savefig(os.path.join(getcwd(), 'path_figures', savefile + str(view_num) + '.png'), dpi=save_dpi)

    # # rotate around
    # axes.view_init(elev=30, azim=150)
    # view_num += 1
    # plt.savefig(os.path.join(getcwd(), 'path_figures', savefile + str(view_num) + '.png'), dpi=save_dpi)
    # fig_path = join(getcwd(), 'path_figures', soln_file + '.html')
    # figure.write_html(fig_path)
    plt.show()

if __name__ == '__main__':
    # python plot_solution.py 0.2 1 1 1
    if argv[1] == '-h':
        print('Example Args:\npython plot_solution.py 0.2 1 1 1')
        print('python plot_solution.py -f pareto_extremes k_1_0_f_1_0 1.5m')
    elif argv[1] == '-d':
        soln_file_input = join(getcwd(), 'ocp_paths', argv[2])
        if argv[2][-1] == 'l': local_input = True
        else: local_input = False
        plot_solution(thrust_limit=1.7, soln_file=soln_file_input, distance=argv[2][:4], local=local_input)
    elif argv[1] == '-f':
        soln_file_input = join(getcwd(), 'ocp_paths', argv[2], argv[3])
        if len(argv) > 4: distance_input = argv[4]
        else: distance_input = '2m'
        local_in = False
        if argv[3][-1] == 'l': local_in = True
        plot_solution(thrust_limit=1.0, soln_file=soln_file_input, distance=distance_input, local=True)

    elif argv[1] == '-c':
        # compare two paths
        if len(argv) > 2: 
            dist = argv[2]
        else:
            dist = '0.5m'
        path1 = join(getcwd(), 'debug', 'all_ccp', dist)
        path2 = join(getcwd(), 'debug', 'all_ccp_compare', dist)
        plot_two_solutions(path1, path2, dist, local=True)
    else:
        if len(argv) > 1: thrust_input = float(argv[1])
        else: thrust_input = 0.2 # float

        if len(argv) > 2: k_weight = argv[2] # string
        else: k_weight = '1'
        if len(argv) > 3: p_weight = argv[3] # string
        else: p_weight = '1'
        if len(argv) > 4: f_weight = argv[4] # string
        else: f_weight = '1'

        soln_dir = 'thrust_test_k_' + k_weight + '_p_' + p_weight + '_f_' + f_weight
        plot_solution(station=True, thrust_limit=thrust_input, soln_dir=soln_dir)
