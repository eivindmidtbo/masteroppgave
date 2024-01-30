"""
Script to be run from jupyter notebook to measure the efficiency of the disk-LSH hash generation using multiprocessing.

10 processess will be run in parallell and each process will be measured using the process time, so that the actual processing time will be measured
"""

from multiprocessing import Pool

# from schemes.disk_lsh import DiskLSH
from schemes.lsh_grid import GridLSH

P_MAX_LON = -8.57
P_MIN_LON = -8.66
P_MAX_LAT = 41.19
P_MIN_LAT = 41.14

R_MAX_LON = 12.53
R_MIN_LON = 12.44
R_MAX_LAT = 41.93
R_MIN_LAT = 41.88

PORTO_DATA = "../data/chosen_data/porto/"
ROME_DATA = "../data/chosen_data/rome/"

# Defining some nescesary variables:

# layers = 4
# diameter = 1.5
# num_disks = 50
# meta_file_p = "../data/chosen_data/porto/META-1000.TXT"
# meta_file_r = "../data/chosen_data/rome/META-1000.TXT"


# Must define some wrapper functions that will be used by the pool processess in the notebook


def fun_wrapper_p_grid(args):
    """Wrapper function for measuring grid hash computation  over porto
    ---
    params : [num_of_files, resolution, layers]
        num_of_files must match one of the meta-files
    """

    num_of_files, resolution, layers = args
    meta_file_p = f"../data/chosen_data/porto/META-{num_of_files}.txt"
    grid = GridLSH(
        "Porto G1",
        P_MIN_LAT,
        P_MAX_LAT,
        P_MIN_LON,
        P_MAX_LON,
        resolution,
        layers,
        meta_file_p,
        PORTO_DATA,
    )

    return grid.measure_hash_computation(1, 1)[0]


def fun_wrapper_r_grid(args):
    """Wrapper function for measuring grid hash computation over rome
    ---
    params : [num_of_files, resolution, layers]
        num_of_files must match one of the meta-files
    """

    num_of_files, resolution, layers = args
    meta_file_r = f"../data/chosen_data/rome/META-{num_of_files}.txt"
    grid = GridLSH(
        "Porto G1",
        R_MIN_LAT,
        R_MAX_LAT,
        R_MIN_LON,
        R_MAX_LON,
        resolution,
        layers,
        meta_file_r,
        ROME_DATA,
    )

    return grid.measure_hash_computation(1, 1)[0]


# All methods takes as input a list: [num_of_files, disks, layers, diameter]


def fun_wrapper_p_naive(args):
    """Wrapper function for measuring disk hash computation
    ---
    params : [num_of_files, disks, layers, diameter]
        num_of_files must match one of the meta-files
    """

    num_of_files, num_disks, layers, diameter = args
    meta_file_p = f"../data/chosen_data/porto/META-{num_of_files}.txt"
    disk = DiskLSH(
        "Porto D1",
        P_MIN_LAT,
        P_MAX_LAT,
        P_MIN_LON,
        P_MAX_LON,
        num_disks,
        layers,
        diameter,
        meta_file_p,
        PORTO_DATA,
    )

    return disk.measure_hash_computation_numerical(1, 1)[0]


def fun_wrapper_p_quadrants(args):
    """Wrapper function for measuring disk hash computation
    ---
    params : [num_of_files, disks, layers, diameter]
        num_of_files must match one of the meta-files
    """

    num_of_files, num_disks, layers, diameter = args
    meta_file_p = f"../data/chosen_data/porto/META-{num_of_files}.txt"
    disk = DiskLSH(
        "Porto D1",
        P_MIN_LAT,
        P_MAX_LAT,
        P_MIN_LON,
        P_MAX_LON,
        num_disks,
        layers,
        diameter,
        meta_file_p,
        PORTO_DATA,
    )

    return disk.measure_hash_computation_with_quad_tree_numerical(1, 1)[0]


def fun_wrapper_p_KD_tree(args):
    """Wrapper function for measuring disk hash computation
    ---
    params : [num_of_files, disks, layers, diameter]
        num_of_files must match one of the meta-files
    """

    num_of_files, num_disks, layers, diameter = args
    meta_file_p = f"../data/chosen_data/porto/META-{num_of_files}.txt"
    disk = DiskLSH(
        "Porto D1",
        P_MIN_LAT,
        P_MAX_LAT,
        P_MIN_LON,
        P_MAX_LON,
        num_disks,
        layers,
        diameter,
        meta_file_p,
        PORTO_DATA,
    )

    return disk.measure_hash_computation_with_KD_tree_numerical(1, 1)[0]


def fun_wrapper_r_naive(args):
    """Wrapper function for measuring disk hash computation
    ---
    params : [num_of_files, disks, layers, diameter]
        num_of_files must match one of the meta-files
    """

    num_of_files, num_disks, layers, diameter = args
    meta_file_r = f"../data/chosen_data/rome/META-{num_of_files}.txt"
    disk = DiskLSH(
        "Rome D1",
        R_MIN_LAT,
        R_MAX_LAT,
        R_MIN_LON,
        R_MAX_LON,
        num_disks,
        layers,
        diameter,
        meta_file_r,
        ROME_DATA,
    )

    return disk.measure_hash_computation_numerical(1, 1)[0]


def fun_wrapper_r_quadrants(args):
    """Wrapper function for measuring disk hash computation
    ---
    params : [num_of_files, disks, layers, diameter]
        num_of_files must match one of the meta-files
    """

    num_of_files, num_disks, layers, diameter = args
    meta_file_r = f"../data/chosen_data/rome/META-{num_of_files}.txt"
    disk = DiskLSH(
        "Rome D1",
        R_MIN_LAT,
        R_MAX_LAT,
        R_MIN_LON,
        R_MAX_LON,
        num_disks,
        layers,
        diameter,
        meta_file_r,
        ROME_DATA,
    )

    return disk.measure_hash_computation_with_quad_tree_numerical(1, 1)[0]


def fun_wrapper_r_KD_tree(args):
    """Wrapper function for measuring disk hash computation
    ---
    params : [num_of_files, disks, layers, diameter]
        num_of_files must match one of the meta-files
    """

    num_of_files, num_disks, layers, diameter = args
    meta_file_r = f"../data/chosen_data/rome/META-{num_of_files}.txt"
    disk = DiskLSH(
        "Rome D1",
        R_MIN_LAT,
        R_MAX_LAT,
        R_MIN_LON,
        P_MAX_LON,
        num_disks,
        layers,
        diameter,
        meta_file_r,
        ROME_DATA,
    )

    return disk.measure_hash_computation_with_KD_tree_numerical(1, 1)[0]
