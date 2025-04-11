import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

def obs_trisurf(meshes, ax, coverage=None, show=True, surface=True, wireframe=True, lw=0.1, alph=0.2):
    """
    plots a list of OBS objects by face
    """
    coverage_idx = 0
    num_faces = np.sum(np.array([len(mesh) for mesh in meshes]))
    n_coverage = len(coverage) if coverage is not None else 0
    highlight_faces = [756,1032] # for debugging purposes
    highlight_faces = []
    # assert num_faces == n_coverage, f'coverage length {n_coverage} does not match number of faces {num_faces}'
    for module_idx, mesh in enumerate(meshes):
        for face in mesh:
            x = [x[0] for x in face]
            y = [x[1] for x in face]
            z = [x[2] for x in face]
            verts = [list(zip(x,y,z))]

            if coverage is not None:
                if coverage[coverage_idx] == 1:
                    if coverage_idx in highlight_faces:
                        pc = Poly3DCollection(verts, fc='tab:purple', alpha=1.0, clip_on=False)
                    else:
                        pc = Poly3DCollection(verts, fc='tab:blue', alpha=alph, clip_on=False)
                else:
                    if coverage_idx in highlight_faces:
                        pc = Poly3DCollection(verts, fc='tab:green', alpha=1.0, clip_on=False)
                    else:
                        pc = Poly3DCollection(verts, fc='red', alpha=alph, clip_on=False)
            else:
                # color faces by module
                if module_idx == 0: pc = Poly3DCollection(verts, fc='tab:blue', alpha=alph, clip_on=False)
                if module_idx == 1: pc = Poly3DCollection(verts, fc='tab:green', alpha=alph, clip_on=False)
                if module_idx == 2: 
                    if (y[0] + y[1] + y[2])/3 < -9.0: pc = Poly3DCollection(verts, fc='tab:green', alpha=alph, clip_on=False)
                    else: pc = Poly3DCollection(verts, fc='tab:purple', alpha=alph, clip_on=False)
                if module_idx == 3: pc = Poly3DCollection(verts, fc='tab:green', alpha=alph, clip_on=False)
                if module_idx == 4: pc = Poly3DCollection(verts, fc='tab:purple', alpha=alph, clip_on=False)
                if module_idx == 5: pc = Poly3DCollection(verts, fc='tab:purple', alpha=alph, clip_on=False)
                if module_idx == 6: pc = Poly3DCollection(verts, fc='tab:purple', alpha=alph, clip_on=False)
                if module_idx == 7: pc = Poly3DCollection(verts, fc='tab:orange', alpha=alph, clip_on=False)
                if module_idx == 8: pc = Poly3DCollection(verts, fc='tab:orange', alpha=alph, clip_on=False)
                if module_idx == 9: pc = Poly3DCollection(verts, fc='tab:green', alpha=alph, clip_on=False)
                # pc = Poly3DCollection(verts, fc='tab:blue', alpha=alph)
            if surface: ax.add_collection3d(pc)
            x.append(x[0])
            y.append(y[0])
            z.append(z[0])
            if wireframe: ax.plot(x,y,z,lw=0.1, c='k', clip_on=False)
            coverage_idx += 1

    # ax.set_aspect('equal')
    if show == True: plt.show()
    return ax

def coverage_area(self, pose, d=0.5, hfov=60, vfov=60):
    """
    get bounds of the coverage area for a given pose
    pose - np.array(x, y, z, pan(ccw around z), tilt(positive up)) assume no roll/swing angle
        > Note: pan is bounded by -pi to pi, tilt goes 0 to pi
    """
    # compute rotation matrices
    R_z = np.array([[np.cos(pose[3]), -np.sin(pose[3]), 0.],
                    [np.sin(pose[3]),  np.cos(pose[3]), 0.],
                    [0.,               0.,              1.]])
    R_x = np.array([[1.,              0.,               0.],
                    [0., np.cos(pose[4]), -np.sin(pose[4])],
                    [0., np.sin(pose[4]),  np.cos(pose[4])]])

    dx = d*np.sin(hfov/2)
    dy = d*np.sin(vfov/2)
    dz = -np.sqrt(d**2 - dx**2 - dy**2)
    tl = R_z @ R_x @ np.array([-dx, dy, dz]) + pose[:3] # top left
    tr = R_z @ R_x @ np.array([dx, dy , dz]) + pose[:3] # top right
    br = R_z @ R_x @ np.array([dx, -dy , dz]) + pose[:3] # bottom right
    bl = R_z @ R_x @ np.array([-dx, -dy , dz]) + pose[:3] # bottom left
    ct = R_z @ R_x @ np.array([0, 0, -d]) + pose[:3] # center
    return tl, tr, br, bl, ct

def plot_cw_opt_path(ax, file='16m_global'):
    dist = file.split('_')[0]
    locality = file.split('_')[1]
    knot_file = dist + '_' + locality + '.csv'
    xfile = file + '_x_interp.csv'
    vfile = file + '_v.csv'
    knots = np.loadtxt(os.path.join(os.getcwd(), 'knot_points', knot_file), delimiter=',')
    x = np.loadtxt(os.path.join(os.getcwd(), 'cw_opt_sol', xfile), delimiter=',')
    v = np.loadtxt(os.path.join(os.getcwd(), 'cw_opt_sol', vfile), delimiter=',') * (float(dist.split('m')[0]) / 16 + 1)
    ax.plot(*[x[:,i] for i in range(3)], 'k-', clip_on=False)
    ax.quiver(*[knots[:,i] for i in range(3)], *[v[:,i] for i in range(3)], color='tab:red', clip_on=False)

    weights = np.linspace(0,1,knots.shape[0])
    # ax.scatter(*[knots[:,i] for i in range(3)], c='tab:orange', edgecolors='k', alpha=1.0, s=50, clip_on=False, zorder=10)
    sc = ax.scatter(*[knots[:,i] for i in range(3)], c=weights, cmap='YlOrBr', edgecolors='k', alpha=1.0, s=30, clip_on=False, zorder=10)

    scbarfs = 15
    scbar = plt.colorbar(sc, shrink=0.6, pad=0.1, location='bottom')
    scbar.ax.tick_params(labelsize=scbarfs)
    scbar.ax.set_xlabel('Normalized Path Progress', fontsize=scbarfs)

    return ax

def plot_packaged_path(folder, file, ax=None, plot_dir=False):
    vgd = file.split('_')[0]

    if ax is None: ax = plt.figure(figsize=(8, 8)).add_subplot(projection='3d')

    # ax = plot_viewpoints(ax, vgd, (file.split('_')[1].split('.')[0] == 'local'))

    filepath = os.path.join(os.getcwd(), folder, file)

    path = np.loadtxt(filepath, delimiter=',')

    ax.plot(path[:,0], path[:,1], path[:,2], 'k-', zorder=10, clip_on=False)

    if plot_dir:
        ax.quiver(path[:,0], path[:,1], path[:,2], path[:,3], path[:,4], path[:,5], length=1, zorder=10, clip_on=False)


    return ax

def normalize(v):
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

def camera_fov_points(pose, viewdir, size=1):
    viewdir += 0.0001 # avoid zero vector
    # define world frame up
    up_world = np.array([0,0,1])
    fov = (np.pi/4, np.pi/4) # horizontal, vertical

    # get camera plane basis vectors
    left = normalize(np.cross(up_world, viewdir))
    up = normalize(np.cross(viewdir, left))

    # get fov adjusted camera plane centered basis vectors
    left = left*np.tan(fov[0]/2)
    up = up*np.tan(fov[1]/2)

    # get viewbox corners
    box_tl = pose + normalize(viewdir)*size + left*size + up*size
    box_tr = pose + normalize(viewdir)*size - left*size + up*size
    box_bl = pose + normalize(viewdir)*size + left*size - up*size
    box_br = pose + normalize(viewdir)*size - left*size - up*size

    # draw camera viewbox lines
    pts = []
    pts.append(pose)
    pts.append(box_tl)
    pts.append(box_tr)
    pts.append(pose)
    pts.append(box_bl)
    pts.append(box_br)
    pts.append(pose)
    pts.append(box_tl)
    pts.append(box_bl)
    pts.append(box_br)
    pts.append(box_tr)

    pts = np.array(pts)
    return pts

def draw_camera(axes, pose, viewdir, size=1):
    """_summary_

    Args:
        axes (_type_): _description_
        pose (_type_): _description_
        viewdir (_type_): _description_
        size (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    viewdir += 0.0001 # avoid zero vector
    # define world frame up
    up_world = np.array([0,0,1])
    fov = (np.pi/4, np.pi/4) # horizontal, vertical

    # get camera plane basis vectors
    left = normalize(np.cross(up_world, viewdir))
    up = normalize(np.cross(viewdir, left))

    # get fov adjusted camera plane centered basis vectors
    left = left*np.tan(fov[0]/2)
    up = up*np.tan(fov[1]/2)

    # get viewbox corners
    box_tl = pose + normalize(viewdir)*size + left*size + up*size
    box_tr = pose + normalize(viewdir)*size - left*size + up*size
    box_bl = pose + normalize(viewdir)*size + left*size - up*size
    box_br = pose + normalize(viewdir)*size - left*size - up*size

    # draw camera viewbox lines
    pts = []
    pts.append(pose)
    pts.append(box_tl)
    pts.append(box_tr)
    pts.append(pose)
    pts.append(box_bl)
    pts.append(box_br)
    pts.append(pose)
    pts.append(box_tl)
    pts.append(box_bl)
    pts.append(box_br)
    pts.append(box_tr)

    pts = np.array(pts)
    axes.plot(pts[:,0], pts[:,1], pts[:,2], 'k-', lw=0.5, clip_on=False, zorder=10)


    return axes


def plot_viewpoints(ax, vgd, local, ordered=True, connect=False):

    # load in knotpoints
    if ordered: filepath = os.path.join( os.getcwd(), '..', 'data', 'ordered_viewpoints', vgd + ('_local' if local else '_global') + '.csv')
    else: filepath = os.path.join( os.getcwd(), '..', 'data', 'coverage_viewpoint_sets', 'coverage_' + vgd + ('_local' if local else '_global') + '_vp_set.csv')
    vps = np.loadtxt(filepath, delimiter=',')

    # if path is ordered, set weights to be normalized path progress
    if ordered: weights = np.linspace(0,1,vps.shape[0])
    else: weights = np.ones(vps.shape[0])

    # plot viewpoints
    sc = ax.scatter(*[vps[:,i] for i in range(4)], c=weights, cmap='YlOrBr', alpha=1.0, s=40, edgecolors='black', clip_on=False, zorder=11)

    # connect viewpoints with dashed lines
    if connect: ax.plot( vps[:,0], vps[:,1], vps[:,2], 'k--', lw=2, clip_on=False, zorder=10)

    for i in range(vps.shape[0]):
        ax = draw_camera(ax, vps[i,:3], vps[i,3:6])

    # 3D colorbar
    scbarfs = 20
    scbar = plt.colorbar(sc, shrink=0.6, pad=0.05, location='bottom')
    scbar.ax.tick_params(labelsize=scbarfs)
    scbar.ax.set_xlabel('Normalized Path Progress', fontsize=scbarfs)

    return ax

def plot_path_direct(folder, file, ax=None, vx=False, ordered=False, local=False):
    if vx: filepath = os.path.join(os.getcwd(), '..', 'data_vx', folder, file)
    else: filepath = os.path.join(os.getcwd(), '..', 'data', folder, file)
    vgd = float(file.split('m')[0])
    # filepath = os.getcwd() + '\\data\\coverage_viewpoint_sets\\' + file
    # for file in os.listdir(dir):
    #     if file.endswith(".npy"): continue
    #     if (cam_dist == float(file[:3])):
    #         if not ((file[5] == 'l') ^ local):
    #             filepath=os.path.join(dir, file)
    # print('found file: ', filepath)
    viewpoints = np.loadtxt(filepath, delimiter=',')
    if ax is None: ax = plt.figure(figsize=(8, 8)).add_subplot(projection='3d')

    # if ordered viewpoints, draw conncting lines
    if ordered: ax.plot(viewpoints[:,0], viewpoints[:,1], viewpoints[:,2], 'k-', lw=2, clip_on=False, zorder=10)
    # ax.scatter(viewpoints[:,0], viewpoints[:,1], viewpoints[:,2], c='k', alpha=0.2)
    # ax.quiver(viewpoints[:,0], viewpoints[:,1], viewpoints[:,2], viewpoints[:,3], viewpoints[:,4], viewpoints[:,5], length=1)
    weights = np.linspace(0,1,viewpoints.shape[0])
    # weights = viewpoints[:,6] / (np.max(viewpoints[:,6]) + 2) + 1/np.max(viewpoints[:,6])


    # if local:
    #     ax.scatter(*[viewpoints[viewpoints[:,6]==0,i] for i in range(3)], c="tab:blue", alpha=0.9, label='Module 0', s=ss)
    #     ax.scatter(*[viewpoints[viewpoints[:,6]==1,i] for i in range(3)], c="tab:orange", alpha=0.9, label='Module 1', s=ss)
    #     ax.scatter(*[viewpoints[viewpoints[:,6]==2,i] for i in range(3)], c="tab:purple", alpha=0.9, label='Module 2', s=ss)
    #     ax.scatter(*[viewpoints[viewpoints[:,6]==3,i] for i in range(3)], c="tab:green", alpha=0.9, label='Module 3', s=ss)
    # else:
    #     ax.scatter(*[viewpoints[:,i] for i in range(3)], c="tab:blue", alpha=0.9, label='_', s=ss)
    # weights = np.ones(viewpoints.shape[0])
    sc = ax.scatter(*[viewpoints[:,i] for i in range(4)], c=weights, cmap='YlOrBr', alpha=1.0, s=40, edgecolors='black', clip_on=False, zorder=10)
    # sc = ax.scatter(*[viewpoints[:,i] for i in range(3)], c=weights, cmap='hsv', alpha=0.9)
    vgd_scale = vgd/16 + 1
    ax.quiver(*[viewpoints[:,i] for i in range(6)], length=vgd_scale, clip_on=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('Viewpoints by Module Membership')
    # ax.legend()
    scbarfs = 15
    scbar = plt.colorbar(sc, shrink=0.6, pad=0.1, location='bottom')
    scbar.ax.tick_params(labelsize=scbarfs)
    scbar.ax.set_xlabel('Normalized Path Progress', fontsize=scbarfs)
    # plt.colorbar(sc, shrink=0.5, label='Path Progression')
    return ax

def save_animation(ax, folder):
    if not os.path.exists(folder): os.mkdir(folder)
    start_time = time.perf_counter()
    time_left = 999
    for i in range(360):
        print('\r', f'Generating frame {i:03d}, {int(time_left)} seconds remaining', end='')
        ax.view_init(elev=10, azim=i)
        plt.savefig(f'./{folder}/{i:03d}.png', dpi=300)
        time_left = (time.perf_counter() - start_time) * (360 - i) / (i + 1)

def get_raw_aoi(sat_bin_count, n_bins):
    x_raw = []
    bin_width = np.pi / 2 / n_bins
    for i in range(n_bins):
        x_raw += [i*bin_width + bin_width/2] * int(sat_bin_count[i])
        
    x_raw += [np.pi] * int(sat_bin_count[-1])
    return x_raw

def set_aspect_equal_3d(axes):
    # get aspect ratios
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()
    zlim = axes.get_zlim()
    axes.set_box_aspect([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])

def get_hex_color_tableau(color):
    tab_colors = {
        'tab:blue' : '#1f77b4',
        'tab:orange' : '#ff7f0e',
        'tab:green' : '#2ca02c',
        'tab:red' : '#d62728',
        'tab:purple' : '#9467bd',
        'tab:brown' : '#8c564b',
        'tab:pink' : '#e377c2',
        'tab:gray' : '#7f7f7f',
        'tab:olive' : '#bcbd22',
        'tab:cyan' : '#17becf'
    }

    if color in tab_colors.keys(): return tab_colors[color]
    else: return None