from station import visualize_station, station_monotone, station_saturation, station_val
from utils import plot_path_direct, save_animation, plot_cw_opt_path, set_aspect_equal_3d, plot_packaged_path, plot_viewpoints, camera_fov_points, get_hex_color_tableau
from matplotlib import pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from saturation_coverage_analysis import compute_coverage_quality, evaluate_good_coverage

def plotly_add_camera_fov(figure, data, step=5, fov_size=1, row=None, col=None):
    # check if plot is a subplot. if so, add camera fov to subplot specified by row and col
    if row is not None and col is not None:
        for i in range(0, data.shape[0], step):
            viewdir_pts = camera_fov_points(data[i,:3], data[i,3:6], size=fov_size)
            figure.add_trace(
                go.Scatter3d(
                    x=viewdir_pts[:,0], y=viewdir_pts[:,1], z=viewdir_pts[:,2],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )

        if data.shape[0] % step != 1:
            i = data.shape[0] - 1
            viewdir_pts = camera_fov_points(data[i,:3], data[i,3:6], size=fov_size)
            figure.add_trace(
                go.Scatter3d(
                    x=viewdir_pts[:,0], y=viewdir_pts[:,1], z=viewdir_pts[:,2],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Camera FOV',
                    showlegend=False
                ),
                row=row, col=col
            )

    else:
        for i in range(0, data.shape[0], step):
            viewdir_pts = camera_fov_points(data[i,:3], data[i,3:6], size=fov_size)
            figure.add_trace(go.Scatter3d(
                x=viewdir_pts[:,0], y=viewdir_pts[:,1], z=viewdir_pts[:,2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

        if data.shape[0] % step != 1:
            i = data.shape[0] - 1
            viewdir_pts = camera_fov_points(data[i,:3], data[i,3:6], size=fov_size)
            figure.add_trace(go.Scatter3d(
                x=viewdir_pts[:,0], y=viewdir_pts[:,1], z=viewdir_pts[:,2],
                mode='lines',
                line=dict(color='black', width=2),
                name='Camera FOV',
                showlegend=False
            ))

    return figure

def plot_dashes_from_knots(knots, fig, dash_length=0.1, row=None, col=None):
    """plot dashes from list of knot points in plotly

    Args:
        knots (_type_): _description_
        fig (_type_): _description_
        dash_length (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    for i in range(len(knots)-1):
        x1, y1, z1 = knots[i,:3]
        x2, y2, z2 = knots[i+1,:3]
        dashes = line_to_dashes_3d(x1, y1, z1, x2, y2, z2, dash_length=dash_length)
        fig = plot_dashes_3d(dashes, fig, row=row, col=col)

    return fig

def line_to_dashes_3d(x1, y1, z1, x2, y2, z2, dash_length=0.1):
    """
    Transforms a 3D line from (x1, y1, z1) to (x2, y2, z2) into dashes with a given dash length.
    
    Parameters:
    - x1, y1, z1: Starting coordinates of the line.
    - x2, y2, z2: Ending coordinates of the line.
    - dash_length: Length of each dash (default is 0.1).
    
    Returns:
    - List of coordinates representing the dash segments in 3D.
    """
    # Calculate the total length of the line in 3D space
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    # Calculate the unit direction vector of the line
    dx = (x2 - x1) / line_length
    dy = (y2 - y1) / line_length
    dz = (z2 - z1) / line_length
    
    # Create the dashes by stepping along the line in 3D space
    dashes = []
    num_dashes = int(line_length // dash_length)
    
    for i in range(0, num_dashes, 3):
        start_x = x1 + i * dash_length * dx
        start_y = y1 + i * dash_length * dy
        start_z = z1 + i * dash_length * dz
        end_x = start_x + 2*dash_length * dx
        end_y = start_y + 2*dash_length * dy
        end_z = start_z + 2*dash_length * dz
        dashes.append((start_x, start_y, start_z, end_x, end_y, end_z))
    
    return dashes

def plot_dashes_3d(dashes, fig, row=None, col=None):
    """
    Plots the dashed line in 3D using the coordinates generated with Plotly.
    
    Parameters:
    - dashes: List of tuples where each tuple contains two points representing a dash segment in 3D.
    """
    if row is not None and col is not None:
        for (start_x, start_y, start_z, end_x, end_y, end_z) in dashes:
            fig.add_trace(
                go.Scatter3d(
                    x=[start_x, end_x],
                    y=[start_y, end_y],
                    z=[start_z, end_z],
                    mode='lines',
                    line=dict(color='black', width=5),
                    showlegend=False
                ),
                row=row, col=col
            )
    else:
        for (start_x, start_y, start_z, end_x, end_y, end_z) in dashes:
            fig.add_trace(go.Scatter3d(
                x=[start_x, end_x],
                y=[start_y, end_y],
                z=[start_z, end_z],
                mode='lines',
                line=dict(color='black', width=5),
                showlegend=False
            ))

    return fig

def plot_path_and_dir(data, figure, savefile=os.path.join('figures','path'), save=False, show=True, fov_size=1, row=None, col=None):
    """plot path and direction in plotly

    Args:
        data (np.ndarray): N x 6 array of [x, y, z, u, v, w] data
        figure (_type_): Plotly figure
        savefile (str, optional): file string to save html of figure to. Defaults to os.path.join('figures','path').
        save (bool, optional): save to savefile?. Defaults to False.
        show (bool, optional): plot figure in browser. Defaults to True.

    Returns:
        _type_: _description_
    """

    if row is not None and col is not None:
        figure.add_trace(
            go.Scatter3d(
                x=data[:,0], y=data[:,1], z=data[:,2],
                mode='lines', name='Path',
                line=dict(color='black', width=6),
                showlegend=False
            ), 
            row=row, col=col
        )
    else:
        figure.add_trace(go.Scatter3d(
            x=data[:,0], y=data[:,1], z=data[:,2],
            mode='lines', name='Path',
            line=dict(color='black', width=6),
            showlegend=False
        ))

    figure = plotly_add_camera_fov(figure, data, step=5, fov_size=fov_size, row=row, col=col)

    figure.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    if save:
        savefile = os.path.join(os.getcwd(), savefile)
        print('saving plot to: ', savefile + '.html')
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        figure.write_html(savefile + '.html')


    if show: figure.show()

    return figure

def plot_final_path(vgd, local, condition='ocp'):
    """plot final path in plotly, save to figure, save to png

    Args:
        vgd (str): viewpoint generation distance (i.e. '2m', '4m', '8m', '16m')
        local (bool): local or global viewpoint generation
        condition (str, optional): path generation method (folder within ./final_paths/ directory). Defaults to 'ocp'.
    """
    figure = station_monotone(
        True,  # convex station model
        title=condition + ': ' + vgd + (' local' if local else ' global') + ' path', 
        save=False, 
        show=False
    )

    # figure = station_monotone(False, title=condition + ': ' + vgd + (' local' if local else ' global') + ' path', save=False, show=False, fig=figure)
    data = np.loadtxt(os.path.join('final_paths', condition, vgd + '_local.csv' if local else vgd + '_global.csv'), delimiter=',')

    # plot knot points
    knotfile = os.path.join('final_paths', 'knots', vgd + '_local.csv' if local else vgd + '_global.csv')
    figure = plot_knots_stripped(knotfile, figure, show=False)

    figure.update_layout(template='simple_white')

    save_file = os.path.join(os.getcwd(), 'figures', 'traj', condition, vgd + ('_local' if local else '_global'))
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    figure = plot_path_and_dir(data, figure, savefile=save_file, save=True, show=False, fov_size=1)

    # save figure to high quality image
    # figure.update_layout(
    #     height=1080,
    #     width=1920
    # )
    figure.write_image(
        save_file + '.png', format='png',
        width=1920,
        height=1080,
        scale=2
    )

def plot_final_path_covg_quality(vgd, local, condition='ocp'):
    """plot station with coverage quality values on mesh in plotly, then save to figures folder

    Args:
        vgd (_type_): _description_
        local (_type_): _description_
        condition (str, optional): _description_. Defaults to 'ocp'.
    """
    # get coverage quality
    quality = compute_coverage_quality(vgd, local, condition)

    # evaluate ratio of good coverage
    good_coverage = evaluate_good_coverage(quality)
    print(vgd, ('local' if local else 'global'), condition, 'good coverage:', good_coverage)

    # plot corresponding saturation values on mesh
    figure = station_val(condition, vgd + ('_local' if local else '_global'), quality, save=False, show=False, annotate_face_index=True)

    # plot knot points
    knotfile = os.path.join('final_paths', 'knots', vgd + '_local.csv' if local else vgd + '_global.csv')
    figure = plot_knots_stripped(knotfile, figure, show=False)

    # remove gridline background
    figure.update_layout(template='simple_white')

    # prepare save directory and file
    save_file = os.path.join(os.getcwd(), 'figures', 'coverage_quality', condition, vgd + ('_local' if local else '_global') + '_covg_quality')
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # load final path data and plot
    data = np.loadtxt(os.path.join('final_paths', condition, vgd + '_local.csv' if local else vgd + '_global.csv'), delimiter=',')
    figure = plot_path_and_dir(data, figure, savefile=save_file, save=True, show=False)

def plot_final_path_saturation_time(vgd, local, condition='ocp'):
    """plot final path with saturation time per mesh displayed on station in plotly, then save to figures folder

    Args:
        vgd (_type_): _description_
        local (_type_): _description_
        condition (str, optional): _description_. Defaults to 'ocp'.
    """
    # plot corresponding saturation values on mesh
    figure = station_saturation(condition, vgd + ('_local' if local else '_global'), 'time', save=False, show=False)

    # plot knot points
    knotfile = os.path.join('final_paths', 'knots', vgd + '_local.csv' if local else vgd + '_global.csv')
    figure = plot_knots_stripped(knotfile, figure, show=False)

    # remove gridline background
    figure.update_layout(template='simple_white')

    # prepare save directory and file
    save_file = os.path.join(os.getcwd(), 'figures', 'traj_time', condition, vgd + ('_local' if local else '_global') + '_time')
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # load final path data and plot
    data = np.loadtxt(os.path.join('final_paths', condition, vgd + '_local.csv' if local else vgd + '_global.csv'), delimiter=',')
    figure = plot_path_and_dir(data, figure, savefile=save_file, save=True, show=False)


def plot_final_path_min_aoi(vgd, local, condition='ocp'):
    """plot final path with min aoi displayed on station in plotly, then save to figures folder

    Args:
        vgd (_type_): _description_
        local (_type_): _description_
        condition (str, optional): _description_. Defaults to 'ocp'.
    """
    # plot corresponding saturation values on mesh
    figure = station_saturation(condition, vgd + ('_local' if local else '_global'), 'min', save=False, show=False)

    # plot knot points
    knotfile = os.path.join('final_paths', 'knots', vgd + '_local.csv' if local else vgd + '_global.csv')
    figure = plot_knots(knotfile, figure, show=False)

    # remove gridline background
    figure.update_layout(template='simple_white')

    # prepare save directory and file
    save_file = os.path.join(os.getcwd(), 'figures', 'traj_min_aoi', condition, vgd + ('_local' if local else '_global') + '_min_aoi')
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # load final path data and plot
    data = np.loadtxt(os.path.join('final_paths', condition, vgd + '_local.csv' if local else vgd + '_global.csv'), delimiter=',')
    figure = plot_path_and_dir(data, figure, savefile=save_file, save=True, show=False)

def plot_covg_quality_study_paths(method='ko_slerp'):
    """generate plots with saturation time station visualization, final study paths, and knot points

    Args:
        method (str, optional): _description_. Defaults to 'ko_slerp'.
    """
    vgds = ['4m', '8m']
    locals = [False, True]
    for vgd in vgds:
        for local in locals:
            plot_final_path_covg_quality(vgd, local, condition='study_paths_' + method)

def plot_saturation_time_study_paths(method='ko_slerp'):
    """generate plots with saturation time station visualization, final study paths, and knot points

    Args:
        method (str, optional): _description_. Defaults to 'ko_slerp'.
    """
    vgds = ['4m', '8m']
    locals = [True, False]
    for vgd in vgds:
        for local in locals:
            plot_final_path_saturation_time(vgd, local, condition='study_paths_' + method)

def plot_min_aoi_study_paths(method='ko_slerp'):
    vgds = ['4m', '8m']
    locals = [True, False]
    for vgd in vgds:
        for local in locals:
            plot_final_path_min_aoi(vgd, local, condition='study_paths_' + method)

def plot_aoi_condition(condition):
    """plot and save final paths in plotly with min aoi data

    Args:
        condition (str): path generation method (folder within ./final_paths/ directory)
    """

    vgds = ['2m', '4m', '8m', '16m']
    locals = [True, False]
    for vgd in vgds:
        for local in locals:
            plot_final_path_min_aoi(vgd, local, condition=condition)

def plot_condition(condition):
    """plot final paths in plotly

    Args:
        condition (str): path generation method (folder within ./final_paths/ directory)
    """

    vgds = ['2m', '4m', '8m', '16m']
    locals = [True, False]
    for vgd in vgds:
        for local in locals:
            try: plot_final_path(vgd, local, condition=condition)
            except FileNotFoundError: print('File not found for', vgd, local, condition)

def plotpath(vgd, local, condition='ocp'):
    local_txt = '_local' if local else '_global'
    print( 'Plotting path for', vgd + local_txt, ' condition:', condition)
    # ax = visualize_station(coverage_file= vgd + '_global_coverage.csv', vx=False)
    ax = visualize_station(coverage_file=None, convex=False, vx=False)
    # ax = visualize_station(coverage_file=None, convex=True, vx=False)
    # ax = visualize_station(coverage_file=None, convex=False, vx=False, original=True, ax=ax)
    # ax = visualize_station(coverage_file=condition + '/' + vgd+local_txt+'_cov.csv', convex=False, vx=False, final=True)

    if condition[0] == 'i':
        soln_folder = 'cw_opt_packaged_' + condition.split('_')[1]
    else: # condition is ocp
        soln_folder = 'pf_final'
    ax = plot_packaged_path(soln_folder, vgd + local_txt + '.csv', ax=ax, plot_dir=True)
    ax = plot_viewpoints(ax, vgd, local, connect=True)
    # ax = plot_path_direct('unfiltered_viewpoints', 'unfiltered_viewpoints_' + vgd + '.csv', ax=ax)
    # ax = plot_path_direct('coverage_viewpoint_sets', 'coverage_' + vgd + '_local_vp_set.csv', ax=ax, ordered=False, local=True)
    # ax.legend()
    # ax = plot_path_direct('ordered_viewpoints', vgd + local_txt + '.csv', ax=ax, ordered=True, local=local)
    # ax = plot_cw_opt_path(ax, vgd + local_txt)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax = plot_path_direct('ordered_viewpoints', vgd + '.csv', ax=ax)
    # plt.savefig('4m_unfiltered.png', dpi=300)
    ax.view_init(elev=30, azim=30)
    ax.dist = 5
    ax.set_axis_off()

    set_aspect_equal_3d(ax)
    # save_animation(ax, vgd + local_txt + '_ocp')


    # dir = os.path.join(os.getcwd(), 'figures', condition)
    # if not os.path.exists(dir): os.makedirs(dir)
    # plt.savefig(os.path.join(os.getcwd(), 'figures', condition, vgd + local_txt + '.png'), dpi=600)
    # plt.savefig(os.path.join(os.getcwd(), 'figures', 'ordered_vps', vgd + local_txt + '.png'), dpi=600)
    # plt.tight_layout()
    plt.show()
    # plt.close('all')

def plot_all_aoi_traj():
    conditions = [
        'ivt_10',
        'ivt_50',
        'ivt_var',
        'ocp_ko',
        'ocp_ko_slerp',
        'ocp_ma',
        'ocp_ma_slerp',
        'ocp_so',
        'ocp_fo'
    ]

    for c in conditions:
        plot_aoi_condition(c)

def plot_file(pathfile, knotfile):
    """plot a path from a knot file and data file

    Args:
        pathfile (_type_): final trajectory file (plot with view direction)
        knotfile (_type_): file of path knots (plot dashed)
    """
    # plot specific path file -- debugging
    figure = station_monotone(True, title=pathfile, save=False, show=False)
    figure = plot_knots(knotfile, figure, show=False)
    data = np.loadtxt(pathfile, delimiter=',')
    plot_path_and_dir(data, figure, save=False, show=True)

def plot_knots_stripped(knotfile, figure, show=False):
    data = np.loadtxt(knotfile, delimiter=',')

    figure.add_trace(go.Scatter3d(
        x=data[:,0], y=data[:,1], z=data[:,2],
        mode='markers',
        marker=dict(
            color=np.arange(data.shape[0])/(data.shape[0]-1),
            colorscale='plasma',
            colorbar=dict(
                orientation='h',
                y=0,
                x=0.5,
                len=0.5,
                tickfont=dict(size=25),
                ticklabelstep=1
            ),
            size=5,
            line=dict(
                width=20,
                color='black'
            ),
            showscale=True
        ),
        # line=dict(dash='solid', color='#424242', width=4),
        showlegend=False
    ))

    figure = plot_dashes_from_knots(data, figure, dash_length=0.3)
    
    figure.update_layout(
        scene_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=-0.07, y=0.15, z=0),
            eye=dict(x=1.25, y=0.8, z=0.9)
        )
    )

    # figure.add_trace(go.Scatter3d(
    #     x=data[:,0], y=data[:,1], z=data[:,2],
    #     mode='lines',
    # ))

    # remove view direction lines
    # figure = plotly_add_camera_fov(figure, data, step=1)

    if show: figure.show()

    return figure

def plot_knots(knotfile, figure, show=False, row=None, col=None):
    data = np.loadtxt(knotfile, delimiter=',')
    if row is not None and col is not None:
        figure.add_trace(
            go.Scatter3d(
                x=data[:,0], y=data[:,1], z=data[:,2],
                mode='markers', name='Knots',
                marker=dict(color='red', size=5)
            ),
            row=row, col=col
        )

        figure.add_trace(
            go.Scatter3d(
                x=data[:,0], y=data[:,1], z=data[:,2],
                mode='lines', name='Knots',
                marker=dict(color='black', size=5)
            ),
            row=row, col=col
        )

    else:
        figure.add_trace(go.Scatter3d(
            x=data[:,0], y=data[:,1], z=data[:,2],
            mode='markers', name='Knots',
            marker=dict(color='red', size=5)
        ))

        figure.add_trace(go.Scatter3d(
            x=data[:,0], y=data[:,1], z=data[:,2],
            mode='lines', name='Knots',
            marker=dict(color='black', size=5)
        ))

    figure = plotly_add_camera_fov(figure, data, step=1, row=row, col=col)

    if show: figure.show()

    return figure

def plot_min_aoi_station_idx(vgd, local, condition='study_paths_fo'):
    # plot corresponding saturation values on mesh
    figure = station_saturation(condition, vgd + ('_local' if local else '_global'), 'min', save=False, show=False, annotate_face_index=True)

    # plot knot points
    knotfile = os.path.join('final_paths', 'knots', vgd + '_local.csv' if local else vgd + '_global.csv')
    figure = plot_knots(knotfile, figure, show=False)

    # remove gridline background
    figure.update_layout(template='simple_white')

    # prepare save directory and file
    save_file = os.path.join(os.getcwd(), 'figures', 'histograms', condition, vgd + ('_local' if local else '_global') + '_min_aoi_face_idx')
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # load final path data and plot
    data = np.loadtxt(os.path.join('final_paths', condition, vgd + '_local.csv' if local else vgd + '_global.csv'), delimiter=',')
    figure = plot_path_and_dir(data, figure, savefile=save_file, save=True, show=False)

def save_station_face_idx_study_paths():
    conditions = [
        'study_paths_ko_slerp',
        'study_paths_fo'
    ]

    vgds = ['4m', '8m']
    locals = [True, False]
    
    for c in conditions:
        for vgd in vgds:
            for local in locals:
                plot_min_aoi_station_idx(vgd, local, condition=c)



if __name__ == "__main__":
    # plot_saturation_time_study_paths('fo')
    # plot_all_aoi_traj()
    # plot_aoi_condition('ocp_fo')
    # plot_aoi_hist_interactive('4m', False, condition='study_paths_fo')
    # save_study_path_histogram_condition('study_paths_fo')
    # save_study_path_aoi_histograms()
    # save_station_face_idx_study_paths()
    # save_all_aoi_overall_hist()
    # save_all_aoi_overall_hist_study()
    plot_condition('study_paths_fo')
    # condition = 'ocp'
    # # vgds = ['2m', '4m', '8m', '16m']
    # vgds = ['4m']
    # locals = [True, False]
    # for vgd in vgds:
    #     for local in locals:
    #         plotpath(vgd, local, condition=condition)
    # # plotpath()
    # save_all_time_overall_hist_study()
    # save_all_time_overall_hist()
    # save_all_aoi_overall_hist_study()
    # save_all_aoi_overall_hist()
    # plot_covg_quality_study_paths(method='fo')