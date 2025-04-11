import numpy as np
import os

def path_length(X):
    """compute the path length for trajectory X

    Args:
        X (np.ndarray): trajectory data (n x 7) - (x, y, z, v0, v1, v2, t) where (v0, v1, v2) is a view direction unit vector
    Returns:
        path_length (float): total path length of trajectory
    """

    path_length = 0

    for i in range(1, X.shape[0]):
        # find distance between points
        d = np.linalg.norm(X[i, :3] - X[i-1, :3])
        path_length += d
    return path_length

def ccp_path_lengths():
    conditions = [
        "2m_global",
        "2m_local",
        "4m_global",
        "4m_local",
        "8m_global",
        "8m_local",
        "16m_global",
        "16m_local"
    ]

    print('Path Length for Ordered Viewpoint Sets:')
    print('condition, path_length (m), N_knot:')
    for c in conditions:
        filepath = os.path.join(os.getcwd(), 'ccp_paths', c + '.csv')

        X = np.loadtxt(filepath, delimiter=',')

        print(f"{c},{path_length(X)},{X.shape[0]}")

if __name__ == "__main__":
    ccp_path_lengths()