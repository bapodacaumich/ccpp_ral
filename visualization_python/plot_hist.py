import os
import numpy as np
import plotly.graph_objects as go
from utils import get_hex_color_tableau
from tqdm import tqdm
from saturation_coverage_analysis import coverage_judgement

# def plot_aoi_hist_interactive(vgd, local, condition='study_paths_fo'):
#     # plot corresponding saturation values on mesh
#     figure = station_saturation(condition, vgd + ('_local' if local else '_global'), 'min', save=False, show=False, side_by_side=True)

#     # plot knot points
#     knotfile = os.path.join('final_paths', 'knots', vgd + '_local.csv' if local else vgd + '_global.csv')
#     figure = plot_knots(knotfile, figure, show=False, row=1, col=1)

#     # remove gridline background
#     figure.update_layout(template='simple_white')

#     # prepare save directory and file
#     save_file = os.path.join(os.getcwd(), 'figures', 'traj_min_aoi', condition, vgd + ('_local' if local else '_global') + '_min_aoi')
#     os.makedirs(os.path.dirname(save_file), exist_ok=True)

#     # load final path data and plot
#     data = np.loadtxt(os.path.join('final_paths', condition, vgd + '_local.csv' if local else vgd + '_global.csv'), delimiter=',')
#     figure = plot_path_and_dir(data, figure, savefile=save_file, save=False, show=False, row=1, col=1)

#     # plot histogram
#     figure = aoi_hist_interactive(vgd, local, condition, figure, row=1, col=2)

#     #launch figure
#     figure.show()

# def aoi_hist_interactive(vgd, local, condition, figure, row=1, col=2):
#     # load in corresponding saturation file
#     saturation_file = os.path.join(os.getcwd(), 'saturation', condition, vgd + ('_local' if local else '_global') + "_sat.csv")
#     saturation = np.loadtxt(saturation_file, delimiter=',')

#     # load min aoi data
#     # aoi_avg = [sat if sat > 0 and sat < np.pi else np.pi for sat in saturation[:,1]]
#     aoi_min = [sat if sat > 0 and sat < np.pi else np.pi for sat in saturation[:,2]]

#     # plot histogram
#     figure.add_trace(
#         go.Histogram(
#             x=aoi_min,
#             marker=dict(color='blue'),
#             showlegend=False
#         ), 
#         row=row, col=col
#     )

#     return figure

def save_all_time_overall_hist():
    conditions = [
        'ivt_10',
        'ivt_50',
        'ivt_var',
        'ocp_fo',
        'ocp_ko',
        'ocp_ko_slerp',
        'ocp_ma',
        'ocp_ma_slerp',
        'ocp_so'
    ]

    vgds = [
        '2m',
        '4m', 
        '8m',
        '16m'
    ]

    locals = [True, False]
    
    for c in conditions:
        for vgd in vgds:
            for local in locals:
                savefile = os.path.join(os.getcwd(), 'figures', 'histograms', c, vgd + ('_local' if local else '_global'), 'time_overall_hist.html')
                os.makedirs(os.path.dirname(savefile), exist_ok=True)
                fig = time_overall_hist(vgd, local, c)
                fig.write_html(savefile)


def save_all_time_overall_hist_study():
    conditions = [
        'study_paths_ko_slerp',
        'study_paths_fo'
    ]

    vgds = ['4m', '8m']
    locals = [True, False]
    
    for c in conditions:
        for vgd in vgds:
            for local in locals:
                savefile = os.path.join(os.getcwd(), 'figures', 'histograms', c, vgd + ('_local' if local else '_global'), 'time_overall_hist.html')
                os.makedirs(os.path.dirname(savefile), exist_ok=True)
                fig = time_overall_hist(vgd, local, c)
                fig.write_html(savefile)

def save_all_aoi_overall_hist():
    conditions = [
        'ivt_10',
        'ivt_50',
        'ivt_var',
        'ocp_fo',
        'ocp_ko',
        'ocp_ko_slerp',
        'ocp_ma',
        'ocp_ma_slerp',
        'ocp_so'
    ]

    vgds = ['2m', '4m', '8m', '16m']
    locals = [True, False]
    
    for c in conditions:
        for vgd in vgds:
            for local in locals:
                savefile = os.path.join(os.getcwd(), 'figures', 'histograms', c, vgd + ('_local' if local else '_global'), 'min_aoi_overall_hist.html')
                os.makedirs(os.path.dirname(savefile), exist_ok=True)
                fig = aoi_overall_hist(vgd, local, c)
                fig.write_html(savefile)


def save_all_aoi_overall_hist_study():
    conditions = [
        'study_paths_ko_slerp',
        'study_paths_fo'
    ]

    vgds = ['4m', '8m']
    locals = [True, False]
    
    for c in conditions:
        for vgd in vgds:
            for local in locals:
                savefile = os.path.join(os.getcwd(), 'figures', 'histograms', c, vgd + ('_local' if local else '_global'), 'min_aoi_overall_hist.html')
                os.makedirs(os.path.dirname(savefile), exist_ok=True)
                fig = aoi_overall_hist(vgd, local, c)
                fig.write_html(savefile)

def time_overall_hist(vgd, local, condition):
    # load in corresponding saturation file
    saturation_file = os.path.join(os.getcwd(), 'saturation', condition, vgd + ('_local' if local else '_global') + "_sat.csv")
    saturation = np.loadtxt(saturation_file, delimiter=',')

    # plot histogram
    fig = go.Figure([
        go.Histogram(
            x=saturation[:,0],
            xbins=dict(start=0),
            showlegend=False
        )]
    )

    fig.update_layout(
        title_text='Time Saturation Distribution'
    )

    return fig


def aoi_overall_hist(vgd, local, condition):
    # load in corresponding saturation file
    saturation_file = os.path.join(os.getcwd(), 'saturation', condition, vgd + ('_local' if local else '_global') + "_sat.csv")
    saturation = np.loadtxt(saturation_file, delimiter=',')

    # load min aoi data
    # aoi_avg = [sat if sat > 0 and sat < np.pi else np.pi for sat in saturation[:,1]]
    aoi_min = [sat if sat > 0 and sat < np.pi else np.pi for sat in saturation[:,2]]


    # plot histogram
    fig = go.Figure([
        go.Histogram(
            x=aoi_min,
            xbins=dict(
                start=0,
                end=np.pi+1e-9
            ),
            showlegend=False
        )]
    )

    fig.update_layout(
        title_text='Min AOI Distribution'
    )

    return fig

def get_raw_aoi(sat_bin_count, n_bins):
    x_raw = []
    bin_width = np.pi / 2 / n_bins
    for i in range(n_bins):
        x_raw += [i*bin_width + bin_width/2] * int(sat_bin_count[i])
        
    x_raw += [np.pi] * int(sat_bin_count[-1])
    return x_raw

def plot_histogram_fig(sat_bin_count, n_bins, titletxt='AOI Distribution', scale=None):

    # x_raw = []
    # bin_width = np.pi / 2 / n_bins
    # for i in range(n_bins):
    #     x_raw += [i*bin_width + bin_width/2] * int(sat_bin_count[i])
        
    # x_raw += [np.pi] * int(sat_bin_count[-1])
    # replace above with below: (haven't tested yet)
    x_raw = get_raw_aoi(sat_bin_count, n_bins)

    nphist = np.histogram(np.array(x_raw) * 180 / np.pi, bins=np.append(np.arange(0, 90, 5), 90.01))

    fig = go.Figure([go.Bar(
        x = nphist[1],
        y = np.array(nphist[0]) / 30,
        width = 0.8 * nphist[1][1],
        marker=dict(color=get_hex_color_tableau('blue')),
        xaxis = 'x',
        offset=0.5
    )])

    # tickv = np.arange(0, np.pi/2 + bin_width, bin_width)
    # tickv = np.append(tickv, np.pi/2 + bin_width)
    # ticktxt = [str(t) for t in tickv]
    # ticktxt[-1] = 'π/2'
    fig.update_layout(
        title_text=titletxt,
        xaxis_title="Angle of Incidence (degrees)",
        yaxis_title="Time (s)",
        xaxis=dict(
            tickvals=np.arange(0, 91, 5),
            ticktext=[str(t) for t in np.arange(0, 91, 5)]
        )
    )

    if scale is not None:
        fig.update_layout(
            yaxis=dict(
                range=[0, scale]
            )
        )

    return fig

def save_all_aoi_histograms_condition_method(vgd, local, method):
    # load saturation binning data
    n_bins = 64
    saturation_bin_count_file = os.path.join(os.getcwd(), 'saturation', method, vgd + ('_local' if local else '_global') + "_sat_" + str(n_bins) + "_bins.csv")
    sat_binning_data = np.loadtxt(saturation_bin_count_file, delimiter=',')

    print('saving histograms for ', method, vgd + ('_local' if local else '_global') + "_sat_" + str(n_bins) + "_bins.csv")

    # for i in tqdm(range(sat_binning_data.shape[0])):
    #     # plot histogram to fig
    #     fig = plot_histogram_fig(sat_binning_data[i], n_bins, titletxt='AOI Distribution Face ' + str(i))

    #     # save histogram
    #     savefile = os.path.join(os.getcwd(), 'figures', 'histograms', method, vgd + ('_local' if local else '_global'), 'by_face', 'histogram_' + str(i) + '.html')
    #     os.makedirs(os.path.dirname(savefile), exist_ok=True)
    #     fig.write_html(savefile)

    # conglomerated faces histogram
    fig = plot_histogram_fig(np.sum(sat_binning_data, axis=0), n_bins, titletxt='AOI Distribution All Faces', scale=20000)

    # save histogram
    savefile = os.path.join(os.getcwd(), 'figures', 'histograms', method, vgd + ('_local' if local else '_global'), 'histogram_all_saturation_time.html')
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    fig.write_html(savefile)

    # conglomerated faces histogram with psychometric curve threshold
    scaled_saturation, _ = coverage_judgement(np.sum(sat_binning_data, axis=0))
    fig = plot_histogram_fig(scaled_saturation, n_bins, titletxt='Psych Scaled AOI Distribution All Faces', scale=20000)

    # save histogram
    savefile = os.path.join(os.getcwd(), 'figures', 'histograms', method, vgd + ('_local' if local else '_global'), 'histogram_psych_scaled_all.html')
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    fig.write_html(savefile)

def save_study_path_aoi_histograms():
    """save histograms associated with study paths

    Args:
        method (_type_): _description_
    """
    conditions = [
        # 'study_paths_ko_slerp',
        'study_paths_fo'
    ]
    vgds = ['4m', '8m']
    locals = [True, False]
    for vgd in vgds:
        for local in locals:
            for c in conditions:
                save_all_aoi_histograms_condition_method(vgd, local, c)

def save_aoi_histograms():
    """save all histograms for a method to html files. not study paths

    Args:
        method (_type_): _description_
    """
    conditions = [
        'ivt_10',
        'ivt_50',
        'ivt_var',
        'ocp_fo',
        'ocp_ko',
        'ocp_ko_slerp',
        'ocp_ma',
        'ocp_ma_slerp',
        'ocp_so'
    ]
    vgds = [
        '2m',
        '4m',
        '8m',
        '16m'
    ]
    locals = [True, False]
    
    for vgd in vgds:
        for local in locals:
            for c in conditions:
                save_all_aoi_histograms_condition_method(vgd, local, c)

if __name__ == "__main__":
    save_study_path_aoi_histograms()