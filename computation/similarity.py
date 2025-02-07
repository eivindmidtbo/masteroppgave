"""
Sheet that will be used to measure the time needed to generate similarities of the disk hashes

Will run N processess in parallell to measure time efficiency
"""

"""
Sheet with collections of methods to generate similarities between trajectories, without having runtime as a focus

"""

import time
import timeit as ti
import pandas as pd
import os, sys

def find_project_root(target_folder="masteroppgave"):
    """Find the absolute path of a folder by searching upward."""
    currentdir = os.path.abspath("__file__")  # Get absolute script path
    while True:
        if os.path.basename(currentdir) == target_folder:
            return currentdir  # Found the target folder
        parentdir = os.path.dirname(currentdir)
        if parentdir == currentdir:  # Stop at filesystem root
            return None
        currentdir = parentdir  # Move one level up

project_root = find_project_root("masteroppgave")

if project_root:
    sys.path.append(project_root)
    print(f"Project root found: {project_root}")
else:
    raise RuntimeError("Could not find 'masteroppgave' directory")


# Imports
from schemes.lsh_bucketing import *
from schemes.lsh_disk import DiskLSH

from schemes.lsh_grid import GridLSH
from schemes.lsh_bucketing import *

from utils.similarity_measures.distance import compute_hash_similarity, compute_hash_similarity_within_buckets, disk_coordinates

from constants import (
    P_MAX_LAT,
    P_MIN_LAT,
    P_MAX_LON,
    P_MIN_LON,
    R_MAX_LAT,
    R_MIN_LAT,
    R_MAX_LON,
    R_MIN_LON,
)

PORTO_CHOSEN_DATA = "../../../dataset/porto/output/"
PORTO_DATA_FOLDER = "../../../dataset/porto/output/"

ROME_CHOSEN_DATA = "../../../dataset/rome/output/"
ROME_DATA_FOLDER = "../../../dataset/rome/output/"



def PORTO_META(size: int):
    return f"{PORTO_DATA_FOLDER}META-{size}.txt"


def ROME_META(size: int):
    return f"{ROME_DATA_FOLDER}META-{size}.txt"




def _constructDisk(
    city: str, diameter: float, layers: int, disks: int, size: int
) -> DiskLSH:
    """Constructs a disk hash object over the given city"""
    if city.lower() == "porto":
        return DiskLSH(
            f"DP_{layers}-{'{:.2f}'.format(diameter)}",
            P_MIN_LAT,
            P_MAX_LAT,
            P_MIN_LON,
            P_MAX_LON,
            disks,
            layers,
            diameter,
            PORTO_META(size),
            PORTO_DATA_FOLDER,
        )
    elif city.lower() == "rome":
        return DiskLSH(
            f"GDR_{layers}-{'{:.2f}'.format(diameter)}",
            R_MIN_LAT,
            R_MAX_LAT,
            R_MIN_LON,
            R_MAX_LON,
            disks,
            layers,
            diameter,
            ROME_META(size),
            ROME_DATA_FOLDER,
        )
    else:
        raise ValueError("City argument must be either porto or rome")

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
            PORTO_DATA_FOLDER,
        )
    elif city.lower() == "rome":
        return GridLSH(
            f"GR_{layers}-{'{:.2f}'.format(res)}",
            R_MIN_LAT,
            R_MAX_LAT,
            R_MIN_LON,
            R_MAX_LON,
            res,
            layers,
            ROME_META(size),
            ROME_DATA_FOLDER,
        )
    else:
        raise ValueError("City argument must be either porto or rome")



def generate_disk_hash_similarity(
    city: str,
    diameter: float,
    layers: int,
    disks: int,
    measure: str = "dtw",
    size: int = 50,
) -> pd.DataFrame:
    """Generates the full disk hash similarities and saves it as a dataframe"""

    Disk = _constructDisk(city, diameter, layers, disks, size)
    hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
    similarities = compute_hash_similarity(
        hashes=hashes, scheme="disk", measure=measure, parallel=True
    )

    return similarities

def generate_grid_hash_similarity(
    city: str, res: float, layers: int, measure: str = "dtw", size: int = 50
) -> pd.DataFrame:
    """Generates the full grid hash similarities and saves it as a dataframe"""

    Grid = _constructGrid(city, res, layers, size)
    hashes = Grid.compute_dataset_hashes()
    similarities = compute_hash_similarity(
        hashes=hashes, scheme="grid", measure=measure, parallel=True
    )

    return similarities


def generate_disk_hash_similarity_coordinates(
    city: str,
    diameter: float,
    layers: int,
    disks: int,
    measure: str = "dtw",
    size: int = 50,
) -> any:
    """Generates the full disk hash similarities and saves it as a dataframe"""

    Disk = _constructDisk(city, diameter, layers, disks, size)
    hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
    hashed_coordinates = disk_coordinates(hashes)
    all_disks_coordinates = Disk.disks

    return hashed_coordinates, all_disks_coordinates

def generate_grid_hash_similarity_coordinates(
    city: str, res: float, layers: int, measure: str = "dtw", size: int = 50
) -> pd.DataFrame:
    """Generates the full grid hash similarities and saves it as a dataframe"""

    Grid = _constructGrid(city, res, layers, size)
    hashes = Grid.compute_dataset_hashes()
    grid_cells = Grid.grid
    return hashes, grid_cells



######################## NEW CODE - BUCKETING ########################

def generate_disk_hash_similarity_with_bucketing(
    city: str,
    diameter: float,
    layers: int,
    disks: int,
    measure: str = "dtw",
    size: int = 50,
) -> pd.DataFrame:
    """
    - Hashes the dataset
    - Places the hashes into buckets
    - Computes the hash similarity values for trajectories within the same bucket and creates a dataframe


    Args:
        city (str): The city to use. Either "porto" or "rome".
        diameter (float): The disks diameter
        layers (int): number of layers in the disk.
        disks (int): number of disks in each layer.
        measure (str, optional): Measure to use. Defaults to "dtw".
        size (int, optional): Number of trajectories to use. Defaults to 50.

    Returns:
        bucketing_system: dict[int, list[str]]: A dictionary containing the bucket system
        pd.DataFrame: The similarity values for the trajectories within the same bucket
    """

    Disk = _constructDisk(city, diameter, layers, disks, size)
    hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
    bucket_system = place_hashes_into_buckets(hashes)
    
    similarities = compute_hash_similarity_within_buckets(
        hashes=hashes, scheme="disk", bucket_system=bucket_system, measure=measure, parallel=True
    )

    return similarities, bucket_system


def generate_grid_hash_similarity_with_bucketing(
    city: str, res: float, layers: int, measure: str = "dtw", size: int = 50
) -> pd.DataFrame:
    """
    - Hashes the dataset
    - Places the hashes into buckets
    - Computes the hash similarity values for trajectories within the same bucket

    Args:
        city (str): The city to use. Either "porto" or "rome".
        res (float): resolution of the grid.
        layers (int): number of layers in the grid.
        measure (str, optional): Measure to use. Defaults to "dtw".
        size (int, optional): Number of trajectories to use. Defaults to 50.

    Returns:
        bucketing_system: dict[int, list[str]]: A dictionary containing the bucket system
        pd.DataFrame: The similarity values for the trajectories within the same bucket
    """

    Grid = _constructGrid(city, res, layers, size) 
    hashes = Grid.compute_dataset_hashes() 
    bucket_system = place_hashes_into_buckets(hashes) 
    
    similarities = compute_hash_similarity_within_buckets(
        hashes=hashes, scheme="grid", bucket_system=bucket_system, measure=measure, parallel=True
    )

    return similarities, bucket_system















# TODO - measure computation time
# def _computeSimilarities(args) -> list:
#     hashes, measure = args
#     elapsed_time = ti.timeit(
#         lambda: MEASURE[measure](hashes), number=1, timer=time.process_time
#     )
#     return elapsed_time


# def measure_disk_hash_similarity_computation_time(
#     city: str,
#     size: int,
#     diameter: float,
#     layers: int,
#     disks: int,
#     hashtype: str,
#     measure: str = "dtw",
#     parallell_jobs: int = 10,
# ) -> list:
#     """
#     Method to measure the execution time of similarity computation of the hashes

#     Param
#     ---
#     city : str
#         Either "porto" or "rome"
#     size : int
#         The dataset-size that will be computed
#     diameter : float
#         The disks diameter
#     layers : int
#         The number of layers that will be used
#     disks : int
#         The number of disks that will be used at each layer
#     hashtype : str
#         "normal" | "quadrants" | "kd"
#     measure : str (Either "ed" or "dtw" - "dtw" default)
#         The measure that will be used for computation
#     parallell_jobs : int
#         The number of jobs that will be run
#     """

#     execution_times = []

#     with Pool(parallell_jobs) as pool:
#         Disk = _constructDisk(city, diameter, layers, disks, size)

#         if measure == "dtw" and hashtype == "kd":
#             hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
#         elif measure == "ed" and hashtype == "normal":
#             hashes = Disk.compute_dataset_hashes()
#         elif measure == "ed" and hashtype == "quadrants":
#             hashes = Disk.compute_dataset_hashes_with_quad_tree()
#         elif measure == "ed" and hashtype == "kd":
#             hashes = Disk.compute_dataset_hashes_with_KD_tree()
#         else:
#             raise ValueError(
#                 "Cannot construct disk hashes as input parameters are uncertain"
#             )

#         time_measurement = pool.map(
#             _computeSimilarities, [(hashes, measure) for _ in range(parallell_jobs)]
#         )
#         execution_times.extend(time_measurement)

#     return execution_times
