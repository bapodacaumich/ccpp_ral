from meshloader import load_meshes, load_vx_station, load_original_stl
from utils import obs_trisurf, set_aspect_equal_3d, get_hex_color_tableau
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def obs_normals(ax, meshes, normals, show=False):
    for i in range(len(meshes)):
        mesh = meshes[i]
        for j in range(len(mesh)):
            face = np.array(mesh[j])
            normal = np.array(normals[i])
            centroid = np.mean(face, axis=0)
            ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], color='r')
    return ax

def visualize_station(show=False, coverage_file='coverage_2m_coverage.csv', show_start=False, vx=False, convex=False, ax=None, final=False, original=False):
    x,y,z = 1.8, 4.7, 2.7
    start = np.array([2, 3, 4.2])
    if vx: meshes, normals = load_vx_station()
    elif original: meshes = load_original_stl()
    else : meshes = load_meshes(convex=convex)
    coverage = None
    if coverage_file is not None:
        if vx: coverage = np.loadtxt(os.path.join(os.getcwd(), '..', 'data_vx', 'coverage_viewpoint_sets', coverage_file)).flatten()
        elif final: 
            coverage = np.genfromtxt(os.path.join(os.getcwd(), 'coverage', coverage_file.split('/')[0], coverage_file.split('/')[1]), delimiter=1,dtype=int)
        else: coverage = np.loadtxt(os.path.join(os.getcwd(), '..', 'data', 'coverage_viewpoint_sets', coverage_file)).flatten()
        # uncoverable = [ 148, 152, 156, 158, 186, 190, 194, 196, 230, 231, 234, 235, 260, 272, 305, 316, 318, 333, 334, 380, 381, 392, 393, 396, 397, 422, 467, 478, 480, 495, 496, 728, 730, 733, 735, 744, 745, 746, 747, 749, 753, 758, 760, 763, 765, 774, 775, 779, 780, 781, 783]
        # for i in range(coverage.shape[0]):
        #     if i in uncoverable: coverage[i] = 1
        # print('Coverage: ', np.sum(coverage)/len(coverage) * 100, '%')
        # idxs = np.arange(len(coverage))
        # print(''.join([str(x) + '\n' for x in idxs[coverage==0]]))

    if original:
        n_coverage = np.sum(np.array([len(mesh) for mesh in meshes]))
        coverage = np.zeros(n_coverage)
    if convex:
        n_coverage = np.sum(np.array([len(mesh) for mesh in meshes]))
        coverage = np.ones(n_coverage)

    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = Axes3D(fig, auto_add_to_figure=False, computed_zorder=False)
        fig.add_axes(ax)
    # ax = obs_normals(ax, meshes, normals, show=False)
    # if convex: coverage = 0 * coverage
    ax = obs_trisurf(meshes, ax, coverage, show=False, alph=0.3)
    # ax = plot_path_direct('station_viewpoints_coverage.csv', ax=ax)
    # ax = obs_normals(ax, meshes, show=False)

    # show axes also
    # ax.plot(x,y,z, 'rx')
    if show_start: ax.plot(start[0], start[1], start[2], 'rx') # start point
    set_aspect_equal_3d(ax)
    if show: plt.show()
    else: return ax

def station_monotone(convex, title='Station', savefile='station', save=False, show=True, fig=None):
    """generate a monotone station visualization with title and save to savefile

    Args:
        convex (bool): which station to plot
        title (string, optional): plot title. Defaults to None.
        savefile (string, optional): savefile with directory. Defaults to None.
        save (bool, optional): _description_. Defaults to False.
        show (bool, optional): _description_. Defaults to True.
    """
    # load in coverage station
    meshes = load_meshes(convex=convex)

    x_by_mesh = []
    y_by_mesh = []
    z_by_mesh = []
    i_by_mesh = []
    j_by_mesh = []
    k_by_mesh = []

    for mesh in meshes:
        xmesh = []
        ymesh = []
        zmesh = []
        imesh = []
        jmesh = []
        kmesh = []

        for idx in range(len(mesh)):
            xmesh.extend([mesh[idx][0][0], mesh[idx][1][0], mesh[idx][2][0]])
            ymesh.extend([mesh[idx][0][1], mesh[idx][1][1], mesh[idx][2][1]])
            zmesh.extend([mesh[idx][0][2], mesh[idx][1][2], mesh[idx][2][2]])
            imesh.append(3*idx)
            jmesh.append(3*idx+1)
            kmesh.append(3*idx+2)

        x_by_mesh.append(xmesh)
        y_by_mesh.append(ymesh)
        z_by_mesh.append(zmesh)
        i_by_mesh.append(imesh)
        j_by_mesh.append(jmesh)
        k_by_mesh.append(kmesh)

    cmeshes = list(itertools.chain(*meshes))

    x = []
    y = []
    z = []
    i = []
    j = []
    k = []


    for idx in range(len(cmeshes)):
        x.extend([cmeshes[idx][0][0], cmeshes[idx][1][0], cmeshes[idx][2][0]])
        y.extend([cmeshes[idx][0][1], cmeshes[idx][1][1], cmeshes[idx][2][1]])
        z.extend([cmeshes[idx][0][2], cmeshes[idx][1][2], cmeshes[idx][2][2]])
        i.append(3*idx)
        j.append(3*idx+1)
        k.append(3*idx+2)

    if fig is None:
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=get_hex_color_tableau('tab:blue'), opacity=0.2, showscale=True)])
    else:
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=get_hex_color_tableau('tab:orange'), opacity=0.2, showscale=True))

    for idx in range(len(x_by_mesh)):
        fig.add_trace(go.Scatter3d(
            x=x_by_mesh[idx], 
            y=y_by_mesh[idx], 
            z=z_by_mesh[idx], 
            # i=i_by_mesh[idx], 
            # j=j_by_mesh[idx], 
            # k=k_by_mesh[idx], 
            mode='lines',
            line=dict(color='black', width=0.3),
            showlegend=False
        ))
    # fig.add_trace(go.Scatter3d(
    #     x=x, y=y, z=z,
    #     mode='lines',
    #     line=dict(color='black', width=0.3)
    # ))

    fig.update_layout(
        scene = dict(
            xaxis=dict(
                title=dict(
                    text='X AXIS'
                ),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(
                    text='Y AXIS'
                ),
                showgrid=False
            ),
            zaxis=dict(
                title=dict(
                    text='Z AXIS'
                ),
                showgrid=False
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title)
    )

    if save:
        savefile = os.path.join(os.getcwd(), savefile + '.html')
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        fig.write_html(savefile)

    if show: fig.show()

    return fig

def station_val(folder, condition, vals, save=False, show=True, title=None, annotate_face_index=False):
    """plotly station saturation visualization

    Args:
        folder (_type_): trajectory generation method
        condition (_type_): vgd and locality conditions for path
        stat (_type_): 'avg', 'min', or 'count'
    """
    ## process values for coloring faces
    # log values for visualization. red is coverage quality below 20 second threshold, blue is above 20 seconds, white is 20 seconds
    saturation_time_threshold = 20
    offset = 1.1 # offset balues to be positive

    # compute log values by offsetting, taking log, then subtracting log threshold + offset
    procvals = vals + offset
    threshold = np.log(saturation_time_threshold + offset)
    procvals = np.log(procvals) - threshold
    zero_ratio = (-np.min(procvals))/(np.max(procvals) - np.min(procvals))
    lowest_val = np.min(procvals[procvals > np.min(procvals)])
    lowest_val_ratio = (lowest_val - np.min(procvals))/(np.max(procvals) - np.min(procvals))

    # load in coverage station
    meshes = load_meshes(convex=False)
    cmeshes = list(itertools.chain(*meshes))

    # triangulate valid faces (get vertices and index order for drawing faces)
    x = []
    y = []
    z = []
    i = []
    j = []
    k = []

    face_idxs = []

    for idx in range(len(cmeshes)):
        x.extend([cmeshes[idx][0][0], cmeshes[idx][1][0], cmeshes[idx][2][0]])
        y.extend([cmeshes[idx][0][1], cmeshes[idx][1][1], cmeshes[idx][2][1]])
        z.extend([cmeshes[idx][0][2], cmeshes[idx][1][2], cmeshes[idx][2][2]])
        i.append(3*idx)
        j.append(3*idx+1)
        k.append(3*idx+2)

        face_idxs.append(str(idx))

    fig = go.Figure()

    if annotate_face_index:
        # draw valid faces
        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, 
                intensity=procvals,
                intensitymode='cell', 
                colorscale=[
                    [0,'rgb(255, 176, 0)'],
                    [lowest_val_ratio, 'rgb(220, 38, 127)'],
                    [zero_ratio, 'rgb(255, 255, 255)'], 
                    [1, 'rgb(100, 143, 255)']
                ],
                showscale=True, 
                customdata=vals,
                hovertemplate='<b>Face Index</b>: %{text} \n <b>Value</b>: %{customdata}',
                text=face_idxs
            )
        )

    else:
        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, 
                intensity=vals,
                intensitymode='cell', 
                colorscale='Viridis', 
                showscale=True,
                hoverinfo='skip'
            )
        )

    fig.update_layout(
        scene = dict(
            xaxis=dict(
                title=dict(
                    text='X AXIS'
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Y AXIS'
                )
            ),
            zaxis=dict(
                title=dict(
                    text='Z AXIS'
                )
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title)
    )

    if save:
        savefile = os.path.join(os.getcwd(), 'figures', folder, condition + '_val_station.html')
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        fig.write_html(savefile)

    if show: fig.show()

    return fig

def station_saturation(folder, condition, stat, save=False, show=True, title=None, side_by_side=False, annotate_face_index=False):
    """plotly station saturation visualization

    Args:
        folder (_type_): trajectory generation method
        condition (_type_): vgd and locality conditions for path
        stat (_type_): 'avg', 'min', or 'count'
    """
    # load in coverage station
    meshes = load_meshes(convex=False)
    cmeshes = list(itertools.chain(*meshes))

    # load in corresponding saturation file
    saturation_file = os.path.join(os.getcwd(), 'saturation', folder, condition + "_sat.csv")
    saturation = np.loadtxt(saturation_file, delimiter=',')

    aoi_avg = [sat if sat > 0 and sat < np.pi else np.pi for sat in saturation[:,1]]
    aoi_min = [sat if sat > 0 and sat < np.pi else np.pi for sat in saturation[:,2]]

    if stat == 'time':
        color_stat = saturation[:,0]
    elif stat == 'avg':
        color_stat = aoi_avg
    elif stat == 'min':
        color_stat = aoi_min

    x = []
    y = []
    z = []
    i = []
    j = []
    k = []

    centroids = []
    face_idxs = []
    for idx in range(len(cmeshes)):
        x.extend([cmeshes[idx][0][0], cmeshes[idx][1][0], cmeshes[idx][2][0]])
        y.extend([cmeshes[idx][0][1], cmeshes[idx][1][1], cmeshes[idx][2][1]])
        z.extend([cmeshes[idx][0][2], cmeshes[idx][1][2], cmeshes[idx][2][2]])
        i.append(3*idx)
        j.append(3*idx+1)
        k.append(3*idx+2)

        centroids.append(np.mean(cmeshes[idx], axis=0).flatten())
        face_idxs.append(str(idx))

    centroids = np.array(centroids)

    if side_by_side:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Minimum Station AOI", "Face AOI Histogram"),
                            specs=[[{'type': 'scene'}, {'type': 'histogram'}]])
    else:
        fig = go.Figure()

    if annotate_face_index:
        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, 
                intensity=color_stat, 
                intensitymode='cell', 
                colorscale='Viridis', 
                showscale=True, 
                hovertemplate='<b>Face Index</b>: %{text}',
                text=face_idxs
            )
        )
    else:
        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, 
                intensity=color_stat, 
                intensitymode='cell', 
                colorscale='Viridis', 
                showscale=True,
                hoverinfo='skip'
            )
        )

    # if annotate_face_index:
    #     fig.add_trace(
    #         go.Scatter3d(
    #             x=centroids[:,0], 
    #             y=centroids[:,1], 
    #             z=centroids[:,2],
    #             mode='markers',
    #             hovertemplate='<b>%{text}</b>',
    #             text=face_idxs,
    #             # textposition='top center',
    #             marker=dict(opacity=0, size=50),
    #             showlegend=False
    #         )
    #     )

    # fig.update_layout(
    #     scene = dict(
    #         xaxis=dict(
    #             title=dict(
    #                 text='X AXIS'
    #             )
    #         ),
    #         yaxis=dict(
    #             title=dict(
    #                 text='Y AXIS'
    #             )
    #         ),
    #         zaxis=dict(
    #             title=dict(
    #                 text='Z AXIS'
    #             )
    #         ),
    #     ),
    #     row=1, col=1
    # )

    # if stat == 'time':
    #     fig.update_layout(
    #         title=dict(text="Face Saturation (seconds): " + folder + "/" + condition)
    #     )
    # elif stat == 'avg':
    #     fig.update_layout(
    #         title=dict(text="Average Station AOI: " + folder + "/" + condition)
    #     )
    # elif stat == 'min':
    #     fig.update_layout(
    #         title=dict(text="Minimum Station AOI: " + folder + "/" + condition)
    #     )

    if not side_by_side:
        fig.update_layout(
            title=dict(text=title)
        )

    if save:
        savefile = os.path.join(os.getcwd(), 'figures', folder, condition + '_' + stat + '_station.html')
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        fig.write_html(savefile)

    if show: fig.show()

    return fig


if __name__ == "__main__":
    visualize_station(show=True, show_start=False)