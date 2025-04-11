from station import station_saturation
from utils import set_aspect_equal_3d
import matplotlib.pyplot as plt

def plot_saturation_ocp(orientation, stat, save=False, show=True):
    folder = 'ocp_' + orientation

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

    for condition in conditions:
        station_saturation(folder, condition, stat, save=save, show=show)

def plot_saturation_cw(tmax, stat, save=False, show=True):
    folder = 'ivt_' + tmax

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

    for condition in conditions:
        station_saturation(folder, condition, stat, save=save, show=show)

def save_all_methods_saturation():
    ocp_methods = [ 'ko', 'so' ]
    stats = ['avg', 'min', 'time']
    ivt_methods = ['var', '10', '50']

    for s in stats:
        for om in ocp_methods:
            plot_saturation_ocp(om, s, save=True, show=False)
            # plot_saturation_ocp(om, s, save=False, show=True)
        for im in ivt_methods:
            plot_saturation_cw(im, s, save=True, show=False)
            # plot_saturation_cw(im, s, save=False, show=True)

if __name__ == '__main__':
    save_all_methods_saturation()