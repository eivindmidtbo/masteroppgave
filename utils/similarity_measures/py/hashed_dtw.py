import numpy as np

from utils.helpers import trajectory_distance as td

# DTW for hashes


def dtw(hash_x: np.ndarray, hash_y: np.ndarray) -> float:
    """
    Computes the edit distance with penalty between two trajectory hashes (Grid | Disk hash)\n
    Runs in layers x O(n^2) time, where n is the length of one hash

    Param
    ---
    hash_x : np array list(list(float))
        The full hash of trajectory x as layers of list of coordinates
    hash_y : np array list(list(str))
        The full hash of trajectory y as layers with list of coordinates

    Returns
    ---
    Their dtw

    Notes
    ---
    Rewritten from https://github.com/bguillouet/traj-dist/blob/master/traj_dist/pydist/edr.py
    """

    x_len = len(hash_x)
    y_len = len(hash_y)

    if x_len != y_len:
        raise ValueError(
            "Number of layers are different for the hashes. Unable to compute edit distance"
        )

    cost = 0

    for layer in range(x_len):
        X = hash_x[layer]
        Y = hash_y[layer]
        X_len = len(X)
        Y_len = len(Y)

        if X_len == 0 or Y_len == 0:
            cost += 0.5
            continue

        M = np.zeros((X_len + 1, Y_len + 1))
        M[1:, 0] = float("inf")
        M[0, 1:] = float("inf")

        for i in range(1, X_len + 1):
            for j in range(1, Y_len + 1):
                s = td.get_euclidean_distance(X[i - 1], Y[j - 1])

                M[i, j] = s + min(M[i][j - 1], M[i - 1][j], M[i - 1][j - 1])
        # print(M)
        cost += float(M[X_len][Y_len])

    return cost
