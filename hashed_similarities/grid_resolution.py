"""
This file contains methods for finding an optimal/working grid resolution / layers

Each experiment will be run in 20 parallell jobs
"""

# Importing nescessary modules

import numpy as np
import pandas as pd
import sys, os
from matplotlib import pyplot as plt
from multiprocessing import Pool

currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# from utils import file_handler as fh
# from utils import metafile_handler as mfh

from utils.helpers import metafile_handler as mfh
from utils.helpers import file_handler as fh
from schemes.helpers.lsh_grid import GridLSH

from utils.similarity_measures.distance import py_edit_distance as py_ed
from utils.similarity_measures.distance import py_edit_distance_penalty as py_edp

# Defining some constants


PORTO_CHOSEN_DATA = "../dataset/porto/output/"
# PORTO_HASHED_DATA = "../hashed_data/grid/porto/"
# PORTO_META_TEST = "../hashed_data/grid/porto/META-50.TXT"
PORTO_META_TEST = "../dataset/porto/output/META-50.TXT"

P_MAX_LON = -8.57
P_MIN_LON = -8.66
P_MAX_LAT = 41.19
P_MIN_LAT = 41.14

ROME_CHOSEN_DATA = "../dataset/rome/output/"
# ROME_HASHED_DATA = "../data/hashed_data/grid/rome/"
# ROME_META_TEST = "../data/hashed_data/grid/rome/META-50.TXT"
ROME_META_TEST = "../dataset/rome/output/META-50.TXT"

R_MAX_LON = 12.53
R_MIN_LON = 12.44
R_MAX_LAT = 41.93
R_MIN_LAT = 41.88

KOLUMBUS_CHOSEN_DATA = "../dataset/kolumbus/output/"
# KOLUMBUS_HASHED_DATA = "../data/hashed_data/grid/kolumbus/"
# KOLUMBUS_META_TEST = "../data/hashed_data/grid/kolumbus/META-50.TXT"
KOLUMBUS_META_TEST = "../dataset/kolumbus/output/META-50.TXT"

K_MAX_LON = 5.80
K_MIN_LON = 5.70
K_MAX_LAT = 59.10
K_MIN_LAT = 58.85

MEASURE = {
    "py_ed": py_ed,
    "py_edp": py_edp,
}

# Defining helper functions:


def _mirrorDiagonal(M: np.ndarray) -> np.ndarray:
    """Flips and mirrors a two-dimenional np.array"""
    return M.values + np.rot90(np.fliplr(M.values))


def _constructGrid(city: str, res: float, layers: int) -> GridLSH:
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
            PORTO_META_TEST,
            PORTO_CHOSEN_DATA,
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
            ROME_META_TEST,
            ROME_CHOSEN_DATA,
        )
    elif city.lower() == "kolumbus":
        return GridLSH(
            f"GK_{layers}-{'{:.2f}'.format(res)}",
            K_MIN_LAT,
            K_MAX_LAT,
            K_MIN_LON,
            K_MAX_LON,
            res,
            layers,
            KOLUMBUS_META_TEST,
            KOLUMBUS_CHOSEN_DATA,
        )
    else:
        raise ValueError("City argument must be either porto or rome")


# True similarities:

P_DTW = _mirrorDiagonal(
    pd.read_csv("../benchmarks/similarities/porto/porto-dtw-testset.csv", index_col=0)
).flatten()  # .stack().values
P_FRE = _mirrorDiagonal(
    pd.read_csv(
        "../benchmarks/similarities/porto/porto-frechet-testset.csv", index_col=0
    )
).flatten()  # .stack().values

R_DTW = _mirrorDiagonal(
    pd.read_csv("../benchmarks/similarities/rome/rome-dtw-testset.csv", index_col=0)
).flatten()  # .stack().values
R_FRE = _mirrorDiagonal(
    pd.read_csv("../benchmarks/similarities/rome/rome-frechet-testset.csv", index_col=0)
).flatten()  # .stack().values

K_DTW = _mirrorDiagonal(
    pd.read_csv(
        "../benchmarks/similarities/kolumbus/kolumbus-dtw-testset.csv", index_col=0
    )
).flatten()  # .stack().values
# K_FRE = _mirrorDiagonal(
#     pd.read_csv(
#         "../benchmarks/similarities/kolumbus/kolumbus-frechet-testset.csv", index_col=0
#     )
# ).flatten()  # .stack().values

# P_dtw_mirrored = mirrorDiagonal(P_dtw).flatten()
# P_fre_mirrored = mirrorDiagonal(P_fre).flatten()

REFERENCE = {
    "portodtw": P_DTW,
    "portofrechet": P_FRE,
    "romedtw": R_DTW,
    "romefrechet": R_FRE,
    "kolumbusdtw": K_DTW,
    # "kolumbusfrechet": K_FRE,
}

DISTANCE_FUNC = {
    "py_ed": "ED",
    "py_edp": "DTW",
}


def _fun_wrapper_corr(args):
    city, res, lay, measure, reference = args
    Grid = _constructGrid(city, res, lay)
    hashes = Grid.compute_dataset_hashes()

    edits = _mirrorDiagonal(MEASURE[measure](hashes)).flatten()
    corr = np.corrcoef(edits, REFERENCE[city.lower() + reference.lower()])[0][1]
    return corr


def _compute_grid_res_layers(
    city: str,
    layers: list[int],
    resolution: list[float],
    measure: str = "py_edp",
    reference: str = "dtw",
    parallell_jobs: int = 20,
):
    """Computations for the visualisation"""

    pool = Pool()

    results = []
    for lay in layers:
        result = []
        for res in np.arange(*resolution):
            print(f"L: {lay}", "{:.2f}".format(res), end="\r")
            # edits = _mirrorDiagonal(MEASURE[measure](hashes)).flatten()

            # corr = np.corrcoef(edits, REFERENCE[city.lower()+reference.lower()])[0][1]
            corrs = pool.map(
                _fun_wrapper_corr,
                [(city, res, lay, measure, reference) for _ in range(parallell_jobs)],
            )
            corr = np.average(np.array(corrs))
            std = np.std(np.array(corrs))
            result.append([corr, res, std])

        results.append([result, lay])

    return results


def plot_grid_res_layers(
    city: str,
    layers: list[int],
    resolution: list[float],
    measure: str = "py_edp",
    reference: str = "dtw",
    parallell_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution and layers for the grid hashes

    Param
    ---
    city : str
        Either "porto" or "rome", throws error unless
    layers : list[int]
        The layers that will be visualised -> [x, y, z...]
    resolution : list[float]
        The resolution that will be visualised -> [min, max, step]
    measure : str (default py_edp)
        The measure that will be used. Either edit distance or dtw -> "py_ed" or "py_edp"
    reference : str (default dtw)
        The true similarities that will be used as reference. Either dtw or frechet
    paralell_jobs : int (default 20)
        Yhe number of parallell jobs that will create the data foundation
    """

    results = _compute_grid_res_layers(
        city, layers, resolution, measure, reference, parallell_jobs
    )

    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    for layer_element in results:
        corrs, layer = layer_element

        corre, res, std = list(zip(*corrs))
        corre = np.array(corre)
        res = np.array(res)
        std = np.array(std)
        ax1.plot(
            res,
            corre,
            c=cmap(float(layer - 1) / (1.2 * N)),
            label=f"{layer} layers",
            lw=2,
        )
        ax2.plot(res, std, c=cmap(float(layer - 1) / (1.2 * N)), ls="dashed")
        # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    ax1.legend(
        loc="lower right",
        ncols=5,
        fontsize=16,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        handletextpad=0.5,
        borderaxespad=0.2,
        columnspacing=1,
    )
    ax2.text(
        0.99,
        0.99,
        f"{city.capitalize()}/{DISTANCE_FUNC[measure]} - {reference.capitalize()}",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=14,
        color="grey",
    )
    ax1.set_xlabel("Grid tile width (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
    ax2.set_ylabel("Standard deviation - Dashed lines", fontsize=18)
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()
