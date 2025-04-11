import numpy as np
from utils import compute_time_intervals, filter_path_na, process_data, load_remeshed_station, align_orientations 
from os.path import join, exists
from os import getcwd, listdir, mkdir
from meshloader import load_meshes
import os


def package_paths(soln_dir=join(getcwd(), 'ocp_paths'),
                  knot_file=join(getcwd(), 'ccp_paths'),
                  save_dir=join(getcwd(), 'packaged_paths'),
                  folder='2m_local'):
    save_dir = join(save_dir, folder)
    soln_dir = join(soln_dir, folder)
    # knot_file = join(knot_file, folder + '.csv')
    # if not exists(save_dir): mkdir(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for file in listdir(soln_dir):
        if file[-5] != 'X': continue
        X = np.loadtxt(join(soln_dir, file), delimiter=' ')
        t = np.loadtxt(join(soln_dir, file[:-5] + 't.csv'), delimiter=' ')
        knots = np.loadtxt(knot_file, delimiter=',')[:,:6]
        knots = filter_path_na(knots)
        data = process_data(knots, X, t)
        np.savetxt(join(save_dir, file[:-6] + '.csv'), data, delimiter=',')

def package_paths_fo(
    soln_dir=join(getcwd(), 'ocp_paths'),
    knot_file=join(getcwd(), 'ccp_paths'),
    save_dir=join(getcwd(), 'packaged_paths_fo'),
    folder='2m_local'
    ):
    """align knot associated path points to point at the same point on the station that the corresponding knot point was pointing at.

    Args:
        soln_dir (_type_, optional): _description_. Defaults to join(getcwd(), 'ocp_paths').
        knot_file (_type_, optional): _description_. Defaults to join(getcwd(), 'ccp_paths').
        save_dir (_type_, optional): _description_. Defaults to join(getcwd(), 'packaged_paths').
        folder (str, optional): _description_. Defaults to '2m_local'.
    """
    # extract vgd and localty from folder name
    vgd = folder.split('_')[-2]
    local = folder.split('_')[-1]

    save_dir = join(save_dir, folder)
    soln_dir = join(soln_dir, folder)
    knot_file = join(knot_file, vgd + '_' + local + '.csv')
    os.makedirs(save_dir, exist_ok=True)
    
    for file in listdir(soln_dir):
        if file[-5] != 'X': continue
        if file[0] != 'k': continue

        X = np.loadtxt(join(soln_dir, file), delimiter=' ')
        t = np.loadtxt(join(soln_dir, file[:-5] + 't.csv'), delimiter=' ')
        knots = np.loadtxt(knot_file, delimiter=',')[:,:6]
        knots = filter_path_na(knots)
        data = align_orientations(knots, X, t, float(vgd[:-1]))
        np.savetxt(join(save_dir, file[:-6] + '.csv'), data, delimiter=',')

def package_all_fo():
    """package all paths in the directory
    """
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

    for c in conditions:
        print(c)
        package_paths_fo(folder=c)

def package_all_cw_opt():
    """package all cw_opt paths in the directory
    """
    condition = [
        '10',
        '50',
        'var'
    ]

    for c in condition:
        package_cw_opt_paths(
            soln_dir=join(getcwd(), 'cw_paths_' + c),
            save_dir=join(getcwd(), 'cw_opt_packaged_' + c)
        )

def package_cw_opt_paths(soln_dir=join(getcwd(), 'cw_paths_var'),
                         knot_file=join(getcwd(), 'ccp_paths'),
                         save_dir=join(getcwd(), 'cw_opt_packaged')
                         ):
    """add orientation data to cw_opt paths using slerp interpolation

    Args:
        soln_dir (_type_, optional): _description_. Defaults to join(getcwd(), 'cw_paths_var').
        knot_file (_type_, optional): _description_. Defaults to join(getcwd(), 'ccp_paths').
        save_dir (_type_, optional): _description_. Defaults to join(getcwd(), 'cw_opt_packaged').
    """
    if not exists(save_dir): mkdir(save_dir)
    
    for file in listdir(soln_dir):
        if file[-12] != 'x': continue

        vgd = file.split('_')[0]
        loc_txt = file.split('_')[1]

        xfile = join(soln_dir, file)
        tfile = join(soln_dir, vgd + '_' + loc_txt + '_t.csv')
        kfile = join(knot_file, vgd + '_' + loc_txt + '.csv')

        X = np.loadtxt(xfile, delimiter=',')
        t = np.loadtxt(tfile, delimiter=',')
        knots = np.loadtxt(kfile, delimiter=',')

        if ( not (knots.shape[0] ==  t.shape[0] + 1)):
            print('Skipping ', join(soln_dir, file), ' due to mismatched knot and time data: ', knots.shape[0], ' vs ', t.shape[0] + 1)
            continue

        idxs = []
        for i in range(knots.shape[0]):
            idxs.append(np.argmin(np.sum((X[:,:3] - knots[i,:3])**2, axis=1)))
            assert idxs[-1] == i*100

        X_new = np.zeros((X.shape[0], 7))
        X_new[:,-1] = X[:,-1]
        X_new[:,:3] = X[:,:3]

        for i in range(1, len(idxs)):
            start_idx = idxs[i-1]
            end_idx = idxs[i]

            n_interp = end_idx - start_idx

            theta = np.arccos(np.dot(knots[i-1, 3:6], knots[i, 3:6]) / (np.linalg.norm(knots[i-1, 3:6]) * np.linalg.norm(knots[i, 3:6]) + 1e-6))
            w = np.linspace(0, 1, n_interp, endpoint=False)

            interp = np.outer(np.sin((1-w)*theta) / np.sin(theta), knots[i-1, 3:6]) + np.outer(np.sin(w*theta) / np.sin(theta), knots[i, 3:6])

            # interp = np.linspace(knots[i-1, 3:6], knots[i, 3:6], n_interp, endpoint=False)

            X_new[start_idx:end_idx, 3:6] = interp

        X_new[-1, 3:6] = knots[-1, 3:6]

        # normalize orientation vectors
        X_new[:, 3:6] /= np.linalg.norm(X_new[:, 3:6], axis=1)[:, None]

        np.savetxt(join(save_dir, vgd + '_' + loc_txt + '.csv'), X_new, delimiter=',')

def moving_average(path, window=5):
    """
    apply moving average to path data
    """
    for i in range(path.shape[0]):
        if i < window:
            path[i, 3:6] = np.mean(path[:i+1, 3:6], axis=0)
        else:
            path[i, 3:6] = np.mean(path[i-window:i+1, 3:6], axis=0)
    return path

def package_moving_average_ocp(
    soln_dir=join(getcwd(), 'ocp_paths'),
    knot_dir=join(getcwd(), 'ccp_paths'),
    save_dir=join(getcwd(), 'packaged_paths_ma')
    ):
    """
    apply moving average to path data
    """

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

    for c in conditions:
        savefolder = join(save_dir, c)
        os.makedirs(savefolder, exist_ok=True)

        # extract vgd and localty from folder name
        vgd = c.split('_')[-2]
        local = c.split('_')[-1]

        # load knot points
        knot_file = join(knot_dir, vgd + '_' + local + '.csv')
        knots = np.loadtxt(knot_file, delimiter=',')

        # filter out path points with NA values
        knots = filter_path_na(knots)[:,:6]

        for file in os.listdir(join(soln_dir, c)):
            if file[-5] != 'X': continue
            if file[0] == 'w': continue
            X = np.loadtxt(join(soln_dir, c, file), delimiter=' ')
            t = np.loadtxt(join(soln_dir, c, file[:-5] + 't.csv'), delimiter=' ')
            # add knot point orientations to path
            data = process_data(knots, X, t)

            # apply moving average to orientation vectors
            data = moving_average(data)

            np.savetxt(join(savefolder, file[:-6] + '.csv'), data, delimiter=',')
            # print(join(savefolder, file[:-6] + '.csv'))


def set_orientations_towards_station(
    soln_dir=join(getcwd(), 'ocp_paths'),
    knot_dir=join(getcwd(), 'ccp_paths'),
    save_dir=join(getcwd(), 'oriented_paths'),
    folder='2m_local'
    ):
    """
    set path orientations towards the closest vertex in the station module that corresponding with the matching knot point (local)
    set path orientations towards the closest vertex in the station model (global)
    """
    # create save directory
    if not exists(save_dir): mkdir(save_dir)
    save_dir = join(save_dir, folder)
    if not exists(save_dir): mkdir(save_dir)

    # extract vgd and localty from folder name
    vgd = folder.split('_')[-2]
    local = folder.split('_')[-1]

    # load knot points
    knot_file = join(knot_dir, vgd + '_' + local + '.csv')
    knots = np.loadtxt(knot_file, delimiter=',')

    # get knot point path point correspondences
    _, knot_idxs = compute_time_intervals(knots, 0.2, 400)

    # # load station model
    # mesh_points_by_module, _ = load_remeshed_station()
    meshes = load_meshes()
    mesh_points_by_module = []
    for module in meshes:
        module_points = []
        for face in module:
            for point in face:
                module_points.append(point)
        mesh_points_by_module.append(np.array(module_points))

    # module membership for local constraint
    module_membership = [0,3,2,3,2,2,2,1,1,3]

    # parse local or global module membership
    mesh_points = []
    if local[0] == 'l':
        mesh_points = [[] for _ in range(max(module_membership)+1)]
        for i in range(len(mesh_points_by_module)):
            mesh_points[module_membership[i]].append(mesh_points_by_module[i])
        for i in range(len(mesh_points)):
            mesh_points[i] = np.concatenate(mesh_points[i], axis=0)
    else:
        mesh_points = np.concatenate(mesh_points_by_module, axis=0)

    if folder.split('_')[0] == 'pf':
        # package all paths in this directory
        for file in listdir(join(soln_dir, folder)):
            # iterate over each path file
            if file[-5] != 'X': continue
            if file[0] == 'w': continue

            # load path and time data
            X = np.loadtxt(join(soln_dir, folder, file), delimiter=' ')
            t = np.loadtxt(join(soln_dir, folder, file[:-5] + 't.csv'), delimiter=' ')

            # ensure path points have corresponding knot points
            assert knot_idxs[-1] < X.shape[0]

            # set up packaged path
            data = np.zeros((X.shape[0], 7))
            data[:, :3] = X[:, :3]
            data[:, -1] = t

            # replace path velocities with orientation vectors
            for i in range(X.shape[0]):
                if local[0] == 'l':
                    # find the knot point idx this point is closest to
                    closest_knot = knots[np.argmin(np.abs(np.array(knot_idxs) - i))]

                    # get relevant module
                    mm = int(np.round(closest_knot[-1]))

                    # find closest vertex in module
                    closest_vertex = mesh_points[mm][np.argmin(np.sum((mesh_points[mm] - X[i,:3])**2, axis=1))]

                else:
                    closest_vertex = mesh_points[np.argmin(np.sum((mesh_points - X[i,:3])**2, axis=1))]

                # get orientation vector
                orientation = closest_vertex - X[i,:3]
                orientation_norm = np.linalg.norm(orientation)
                n_orientation = orientation / orientation_norm

                # set orientation vector
                data[i, 3:6] = n_orientation

            np.savetxt(join(save_dir, file[:-6] + '.csv'), data, delimiter=',')


if __name__ == "__main__":
    # package_data()
    # package_paths(folder='2m_local')
    # package_paths(folder='2m_global')
    # package_paths(folder='4m_local')
    # package_paths(folder='4m_global')
    # package_paths(folder='8m_local')
    # package_paths(folder='8m_global')
    # package_paths(folder='16m_local')
    # package_paths(folder='16m_global')
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '2m_global.csv'), folder='pf_2m_global', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '2m_local.csv'), folder='pf_2m_local', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '4m_global.csv'), folder='pf_4m_global', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '4m_local.csv'), folder='pf_4m_local', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '8m_global.csv'), folder='pf_8m_global', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '8m_local.csv'), folder='pf_8m_local', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '16m_global.csv'), folder='pf_16m_global', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_paths(knot_file=join(getcwd(), 'ccp_paths', '16m_local.csv'), folder='pf_16m_local', save_dir=join(getcwd(), 'packaged_paths_ko_slerp'))
    # package_all_cw_opt()
    package_all_fo()
    # set_orientations_towards_station(folder='pf_2m_global')
    # set_orientations_towards_station(folder='pf_2m_local')
    # set_orientations_towards_station(folder='pf_4m_global')
    # set_orientations_towards_station(folder='pf_4m_local')
    # set_orientations_towards_station(folder='pf_8m_global')
    # set_orientations_towards_station(folder='pf_8m_local')
    # set_orientations_towards_station(folder='pf_16m_global')
    # set_orientations_towards_station(folder='pf_16m_local')
    # package_moving_average_ocp()