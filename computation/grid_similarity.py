""" 
Sheet that will be used to measure the time needed to generate similarities of the grid-hashes 

Will run N processess in parallell to measure time efficiency
"""

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


from schemes.lsh_grid import GridLSH

from utils.similarity_measures.distance import (
    compute_hash_similarity,
)

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

PORTO_CHOSEN_DATA = "../../dataset/porto/output/"
PORTO_DATA_FOLDER = "../../dataset/porto/output/"

ROME_CHOSEN_DATA = "../../dataset/rome/output/"
ROME_DATA_FOLDER = "../../dataset/rome/output/"


def PORTO_META(size: int):
    return f"../{PORTO_DATA_FOLDER}/META-{size}.txt"


def ROME_META(size: int):
    return f"../{ROME_DATA_FOLDER}/META-{size}.txt"

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
  

def generate_grid_hash_similarity_coordinates(
    city: str, res: float, layers: int, measure: str = "dtw", size: int = 50
) -> pd.DataFrame:
    """Generates the full grid hash similarities and saves it as a dataframe"""

    Grid = _constructGrid(city, res, layers, size)
    hashes = Grid.compute_dataset_hashes()
    grid_cells = Grid.grid
    return hashes, grid_cells
