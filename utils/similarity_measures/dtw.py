""" Sheet containing DTW methods related to true similarity creation """

import numpy as np
import pandas as pd
import collections as co

from traj_dist.pydist.dtw import e_dtw as p_dtw
from traj_dist.distance import dtw as c_dtw

from multiprocessing import Pool
import timeit as ti
import time


def py_dtw(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """ 
    Method for computing DTW similarity between all trajectories in a given dataset using python. 

    Params
    ---
    trajectories : dict[str, list[list[float]]]
        A dictionary containing the trajectories

    Returns
    ---
    A nxn pandas dataframe containing the pairwise similarities - sorted alphabetically 
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectoris = len(sorted_trajectories)

    M = np.zeros((num_trajectoris, num_trajectoris))
    
    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            X = np.array(sorted_trajectories[traj_i])
            Y = np.array(sorted_trajectories[traj_j])
            dtw = p_dtw(X,Y)
            M[i,j] = dtw
            if i == j: 
                break
    
    df = pd.DataFrame(M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys())

    return df


def measure_py_dtw(args):
    """ Method for measuring time efficiency using py_dtw 

    Params
    ---
    args : (trajectories: dict[str, list[list[float]]], number: int, repeat: int) list
    """
    trajectories, number, repeat = args

    measures = ti.repeat(lambda: py_dtw(trajectories), number=number, repeat=repeat, timer=time.process_time)
    return measures



def cy_dtw(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """ 
    Method for computing DTW similarity between all trajectories in a given dataset using cython. 

    Params
    ---
    trajectories : dict[str, list[list[float]]]
        A dictionary containing the trajectories

    Returns
    ---
    A nxn pandas dataframe containing the pairwise similarities - sorted alphabetically 
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectoris = len(sorted_trajectories)

    M = np.zeros((num_trajectoris, num_trajectoris))
    
    for i, traj_i in enumerate(sorted_trajectories.keys()):
        for j, traj_j in enumerate(sorted_trajectories.keys()):
            X = np.array(sorted_trajectories[traj_i])
            Y = np.array(sorted_trajectories[traj_j])
            dtw = c_dtw(X,Y)
            M[i,j] = dtw
            if i == j: 
                break
    
    df = pd.DataFrame(M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys())

    return df

def measure_cy_dtw(args):
    """ Method for measuring time efficiency using py_dtw """
    
    trajectories, number, repeat = args

    measures = ti.repeat(lambda: cy_dtw(trajectories), number=number, repeat=repeat, timer=time.process_time)
    return measures



# Helper function for dtw parallell programming for speedy computations
def _fun_wrapper(args):
        x,y,j = args
        dtw = c_dtw(x,y)
        return dtw, j

def cy_dtw_pool(trajectories: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Same as above, but using a pool of procesess for speedup
    """

    sorted_trajectories = co.OrderedDict(sorted(trajectories.items()))
    num_trajectoris = len(sorted_trajectories)

    M = np.zeros((num_trajectoris, num_trajectoris))  
        
    pool = Pool(12)

    for i, traj_i in enumerate(sorted_trajectories.keys()):

        dtw_elements = pool.map(_fun_wrapper, [(np.array(sorted_trajectories[traj_i]), np.array(sorted_trajectories[traj_j]), j) for j, traj_j in enumerate(sorted_trajectories.keys()) if i >= j])

        for element in dtw_elements:
            M[i,element[1]] = element[0]

    df = pd.DataFrame(M, index=sorted_trajectories.keys(), columns=sorted_trajectories.keys())
    
    return df


#DTW for hashes

def dtw_hash(hash_x: np.ndarray, hash_y: np.ndarray) -> float:
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
        raise ValueError("Number of layers are different for the hashes. Unable to compute edit distance")

    cost = 0


    for layer in range(x_len):
        X = hash_x[layer]
        Y = hash_y[layer]
        X_len = len(X)
        Y_len = len(Y)

        if (X_len == 0 or Y_len == 0):
            cost += 0.5
            continue

        M = np.zeros((X_len + 1, Y_len + 1))
        M[1:, 0] = float('inf')
        M[0, 1:] = float('inf')

        for i in range(1, X_len + 1):
            for j in range(1, Y_len + 1):
                s = td.get_euclidean_distance(X[i-1], Y[j-1])

                M[i,j] = s + min(M[i][j-1], M[i-1][j], M[i-1][j-1])
        #print(M)
        cost += float(M[X_len][Y_len])
    
    return cost


