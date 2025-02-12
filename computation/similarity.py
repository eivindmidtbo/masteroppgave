"""
Sheet that will be used to measure the time needed to generate similarities of the disk hashes

Will run N processess in parallell to measure time efficiency
"""

"""
Sheet with collections of methods to generate similarities between trajectories, without having runtime as a focus

"""
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
    # print(f"Project root found: {project_root}")
else:
    raise RuntimeError("Could not find 'masteroppgave' directory")

# Imports
import pandas as pd
from schemes.lsh_disk import DiskLSH
from schemes.lsh_grid import GridLSH
from schemes.lsh_bucketing import *
import timeit as ti
import time

from utils.similarity_measures.hashed_dtw import cy_dtw_hashes, cy_dtw_hashes_pool
from utils.similarity_measures.hashed_frechet import cy_frechet_hashes, cy_frechet_hashes_pool

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

PORTO_DATA_FOLDER = f"../../../dataset/porto/output/"
ROME_DATA_FOLDER = f"{project_root}/dataset/rome/output/"

#Get metafile from porto
def PORTO_META(size: int):
    return f"{PORTO_DATA_FOLDER}META-{size}.txt"

#Get metafile from rome
def ROME_META(size: int):
    return f"{ROME_DATA_FOLDER}META-{size}.txt"

def transform_np_numerical_disk_hashes_to_non_np(
    hashes: dict[str, list[list[float]]]
) -> dict[str, list[list[list[float]]]]:
    """Transforms the numerical disk hashes to a format that fits the true dtw similarity measure (non numpy input)"""
    transformed_data = {
        key: [[array.tolist() for array in sublist] for sublist in value]
        for key, value in hashes.items()
    }
    return transformed_data

def disk_coordinates(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """The hashed disk coordinates"""
    hashed_coordinates = transform_np_numerical_disk_hashes_to_non_np(hashes)
    return hashed_coordinates

def _constructDisk(city: str, diameter: float, layers: int, disks: int, size: int) -> DiskLSH:
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
    
def compute_hash_similarity(
    hashes: dict[str, list[list[list[float]]]],
    scheme: str,
    measure: str,
    parallel: bool = False,
) -> pd.DataFrame:
    """
    Computes the similarity between the hashes for both schemes with the given measure

    Args:
        parallel (bool, optional): Rather to speed up computation. Defaults to False.

    Returns:
        pd.DataFrame: similarities between the hashes
    """
    if scheme == "disk":
        hashes = transform_np_numerical_disk_hashes_to_non_np(hashes)
    # NOTE - if scheme =="grid" then the hashes are already in the correct format, I.E non numpy
    if measure == "dtw":
        if parallel:
            return cy_dtw_hashes_pool(hashes)
        else:
            return cy_dtw_hashes(hashes)
    elif measure == "frechet":
        if parallel:
            return cy_frechet_hashes_pool(hashes)
        else:
            return cy_frechet_hashes(hashes)

def generate_disk_hash_similarity(
    city: str,
    diameter: float,
    layers: int,
    disks: int,
    measure: str = "dtw",
    size: int = 50,
) -> pd.DataFrame:
    """Generates the full disk hash similarities and saves it as a dataframe (wrapper function)"""

    Disk = _constructDisk(city, diameter, layers, disks, size)
    hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
    similarities = compute_hash_similarity(
        hashes=hashes, scheme="disk", measure=measure, parallel=True
    )

    return similarities

def generate_grid_hash_similarity(
    city: str, res: float, layers: int, measure: str = "dtw", size: int = 50
) -> pd.DataFrame:
    """Generates the full grid hash similarities and saves it as a dataframe (wrapper function)"""

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
    print("Generating disk hash similarity with bucketing")
    Disk = _constructDisk(city, diameter, layers, disks, size)
    print("Computing dataset hashes")
    hashes = Disk.compute_dataset_hashes_with_KD_tree_numerical()
    print("Placing hashes into buckets")
    bucket_system = place_hashes_into_buckets_individual(hashes)
    print("Computing hash similarity within buckets")
    similarities = compute_hash_similarity_within_buckets(
        hashes=hashes, scheme="disk", bucket_system=bucket_system, measure=measure, parallel=False
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
        hashes=hashes, scheme="grid", bucket_system=bucket_system, measure=measure, parallel=False
    )

    return similarities, bucket_system

def compute_hash_similarity_within_buckets(
    hashes: dict[str, list[list[list[float]]]],
    scheme: str,
    measure: str,
    bucket_system: dict[int, list[str]],
    parallel: bool = False,
) -> pd.DataFrame:
    """
    Computes the similarity between the hashes in the same bucket for all buckets
    
    Returns:
        pd.DataFrame: A global similarity matrix for all trajectories
    """
    
    
    # Get all trajectory names across all buckets
    all_trajectories = set()
    
    for bucket_trajectories in bucket_system.values():
        all_trajectories.update(bucket_trajectories)

    # Convert set to a sorted list for a stable DataFrame index
    all_trajectories = sorted(all_trajectories)

    # Create a DataFrame initialized with zeros
    global_similarity_matrix = pd.DataFrame(
        0,
        index=all_trajectories,
        columns=all_trajectories,
        dtype=float,
    )

    # Compute similarity matrices for each bucket
    for key in bucket_system:
        # Skip buckets with only one trajectory
        if len(bucket_system[key]) <= 1:
            continue

        # Filter hashes for the current bucket
        bucket_hashes = {file: hashes[file] for file in bucket_system[key]}
        
        # Transform hashes if necessary
        if scheme == "disk":
            bucket_hashes = transform_np_numerical_disk_hashes_to_non_np(bucket_hashes)

        # Compute similarities within the current bucket
        if measure == "dtw":
            similarities = (
                cy_dtw_hashes_pool(bucket_hashes) if parallel else cy_dtw_hashes(bucket_hashes)
            )
        elif measure == "frechet":
            similarities = (
                cy_frechet_hashes_pool(bucket_hashes) if parallel else cy_frechet_hashes(bucket_hashes)
            )

        # Create a DataFrame for the current bucket
        trajectory_names = list(bucket_hashes.keys())
        similarity_df = pd.DataFrame(similarities, index=trajectory_names, columns=trajectory_names)

        # Update the global similarity matrix with values from the current bucket
        for i, traj_i in enumerate(trajectory_names):
            for j, traj_j in enumerate(trajectory_names):
                global_similarity_matrix.loc[traj_i, traj_j] = max(
                    global_similarity_matrix.loc[traj_i, traj_j], similarity_df.iloc[i, j]
                )

    return global_similarity_matrix


def measure_hashed_cy_bucketing(
    hashes: dict[str, list[list[list[float]]]],
    scheme: str,
    measure: str,
    bucket_system: dict[int, list[str]],
    parallel: bool = False):
    
    """Method for measuring time efficiency using cy_dtw_hashes"""
        
    measures = ti.repeat(
        lambda: compute_hash_similarity_within_buckets(hashes=hashes, scheme=scheme, measure=measure, bucket_system=bucket_system, parallel=parallel)
                ,number=1, repeat=1, timer=time.process_time
    )
    
    return measures
