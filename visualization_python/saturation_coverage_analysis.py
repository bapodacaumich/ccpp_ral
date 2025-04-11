import numpy as np
import os
from utils import get_raw_aoi

def load_saturation_data(vgd, local, method, n_bins=64):
    """load saturation bin counts

    Args:
        vgd (str): which VGD to load
        local (bool): whether to load local or global saturation data
        method (str): which method to load

    Returns:
        np.ndarray: bin counts
    """
    # load saturation binning data
    saturation_bin_count_file = os.path.join(os.getcwd(), 'saturation', method, vgd + ('_local' if local else '_global') + "_sat_" + str(n_bins) + "_bins.csv")
    sat_binning_data = np.loadtxt(saturation_bin_count_file, delimiter=',')
    return sat_binning_data

def coverage_judgement(saturation, crossover=75, slope=0.2, n_bins=64):
    """compute coverage heuristic for this face

    Args:
        saturation (np.ndarray): aoi counts per 1/30 seconds for a single face
        threshold (float): threshold (0-1)

    Returns:
        float: quality of coverage
    """
    bin_width = np.pi / 2 / n_bins
    deg_vals = np.arange(0, bin_width * n_bins, bin_width) * 180 / np.pi + bin_width / 2

    # use psychometric curve at 80 degrees
    psych_vals = 1 / (1 + np.exp(slope * (deg_vals - crossover)))

    # scale values according to psychometric curve, ignore last bin (overflow)
    scaled_saturation = saturation[:n_bins] * psych_vals

    return scaled_saturation, bin_width

    # # numerical integration
    # quality = np.sum(scaled_saturation) * bin_width

    # return quality

def compute_coverage_quality(vgd, local, method):
    """compute coverage quality for each face individually

    Args:
        vgd (str): which VGD to load
        local (bool): whether to load local or global saturation data
        method (str): which method to load

    Returns:
        np.ndarray: coverage qualities
    """
    sat = load_saturation_data(vgd, local, method)

    qualities = []
    for face in sat:
        scaled_saturation, bin_width = coverage_judgement(face)
        quality = np.sum(scaled_saturation)
        qualities.append(quality)
    qualities = np.array(qualities)

    # zero_idx = np.where(qualities == 0)[0]
    # qualities[zero_idx] = -1
    # quali
    uncoverable = [ 148, 152, 156, 158, 186, 190, 194, 196, 218, 230, 231, 234, 235, 260, 272, 305, 316, 318, 333, 334, 380, 381, 392, 393, 396, 397, 422, 467, 478, 480, 495, 496, 728, 730, 733, 735, 744, 745, 746, 747, 749, 753, 758, 760, 763, 765, 774, 775, 779, 780, 781, 783 ]
    qualities[uncoverable] = -1

    return qualities

def evaluate_good_coverage(qualities, threshold=20):
    """compute ratio of faces with coverage quality metric above 20 seconds of good AOI

    Args:
        qualities (_type_): coverage quality for each face
        threshold (int, optional): coverage quality threshold (seconds). Defaults to 20.

    Returns:
        good_coverage_ratio (float): ratio of faces with good coverage
    """
    uncoverable = [ 148, 152, 156, 158, 186, 190, 194, 196, 218, 230, 231, 234, 235, 260, 272, 305, 316, 318, 333, 334, 380, 381, 392, 393, 396, 397, 422, 467, 478, 480, 495, 496, 728, 730, 733, 735, 744, 745, 746, 747, 749, 753, 758, 760, 763, 765, 774, 775, 779, 780, 781, 783 ]

    # set up numerator and denominator of coverage quality threshold ratio
    num = 0
    denom = 0

    # accumulate numerator and denominator of coverage quality threshold ratio and ignore uncoverable (invalid) faces
    for i in range(qualities.shape[0]):
        if i in uncoverable: continue
        denom += 1

        # ignore uncoverable faces
        if qualities[i] > threshold:
            num += 1

    return num / denom

# if __name__ == "__main__":
#     compute_coverage_quality('8m', False, 'study_paths_fo')