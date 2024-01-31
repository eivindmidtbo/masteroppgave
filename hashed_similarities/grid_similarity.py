""" 
Sheet that will be used to measure the time needed to generate similarities of the grid-hashes 

Will run N processess in parallell to measure time efficiency
"""

from multiprocessing import Pool
import time
import timeit as ti
import pandas as pd
import os, sys

currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from schemes.helpers.lsh_grid import GridLSH

from utils.similarity_measures.distance import py_edit_distance as py_ed
from utils.similarity_measures.distance import py_edit_distance_penalty as py_edp
from utils.similarity_measures.distance import (
    py_edit_distance_penalty_parallell as py_edp_parallell,
)

P_MAX_LON = -8.57
P_MIN_LON = -8.66
P_MAX_LAT = 41.19
P_MIN_LAT = 41.14

R_MAX_LON = 12.53
R_MIN_LON = 12.44
R_MAX_LAT = 41.93
R_MIN_LAT = 41.88

PORTO_CHOSEN_DATA = "../dataset/porto/output/"
# PORTO_HASHED_DATA = "../hashed_data/grid/porto/"

# ROME_CHOSEN_DATA = "../data/chosen_data/rome/"
# ROME_HASHED_DATA = "../data/hashed_data/grid/rome/"


def PORTO_META(size: int):
    # return f"../hashed_data/grid/porto/META-{size}.txt"
    return f"../dataset/porto/output/META-{size}.txt"


# def ROME_META(size: int): return f"../data/hashed_data/grid/rome/META-{size}.TXT"

MEASURE = {
    # "ed" : py_ed,
    "dtw": py_edp,
}


def _constructGrid(city: str, res: float, layers: int, size: int) -> GridLSH:
    """Constructs a grid hash object over the given city"""
    if city.lower() == "porto":
        return GridLSH(
            f"GP_{layers}-{'{:.2f}'.format(res)}",
            P_MIN_LAT,
            P_MAX_LAT,
            P_MIN_LON,
            P_MAX_LON,
            res,
            layers,
            PORTO_META(size),
            PORTO_CHOSEN_DATA,
        )
    # elif city.lower() == "rome":
    #     return GridLSH(f"GR_{layers}-{'{:.2f}'.format(res)}", R_MIN_LAT, R_MAX_LAT, R_MIN_LON, R_MAX_LON, res, layers, ROME_META(size), ROME_CHOSEN_DATA)
    else:
        raise ValueError("City argument must be either porto or rome")


def _computeSimilarities(args) -> list:
    hashes, measure = args
    elapsed_time = ti.timeit(
        lambda: MEASURE[measure](hashes), number=1, timer=time.process_time
    )
    return elapsed_time


def measure_grid_hash_similarity_computation_time(
    city: str,
    size: int,
    res: float,
    layers: int,
    measure: str = "dtw",
    parallell_jobs: int = 10,
) -> list:
    """
    Method to measure the execution time of similarity computation of the hashes

    Param
    ---
    city : str
        Either "porto" or "rome"
    size : int
        The dataset-size that will be computed
    res : float
        The grid resolution
    layers : int
        The number of layers that will be used
    measure : str (Either "ed" or "dtw" - "dtw" default)
        The measure that will be used for computation
    parallell_jobs : int
        The number of jobs that will be run
    """
    times = []
    with Pool(parallell_jobs) as pool:
        Grid = _constructGrid(city, res, layers, size)
        hashes = Grid.compute_dataset_hashes()
        time_measurement = pool.map(
            _computeSimilarities, [(hashes, measure) for _ in range(parallell_jobs)]
        )
        times.extend(time_measurement)
    return times


def generate_grid_hash_similarity(city: str, res: float, layers: int) -> pd.DataFrame:
    """Generates the full grid hash similarities and saves it as a dataframe"""

    Grid = _constructGrid(city, res, layers, 50)
    hashes = Grid.compute_dataset_hashes()
    similarities = py_edp_parallell(hashes)

    return similarities
