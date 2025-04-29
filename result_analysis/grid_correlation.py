"""
This file contains methods for finding an optimal/working number of resolution and layers for grid scheme. 

Each experiment will be run in 20 parallell jobs
"""

# Importing nescessary modules

import numpy as np
import sys, os
from matplotlib import pyplot as plt
from multiprocessing import Pool
import warnings

# Suppress only invalid value warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import pandas as pd
from computation.similarity import generate_grid_hash_similarity
from constants import COLOR_MAP

# Defining helper functions:
def mirrorDiagonal(M: np.ndarray) -> np.ndarray:
    """Flips and mirrors a two-dimenional np.array"""
    return M.values + np.rot90(np.fliplr(M.values))


# Brukes av alle funksjonene
def _fun_wrapper_corr(args):
    
    city, res, layers, true_sim_matrix, measure, size = args
    
    hashed_similarities = generate_grid_hash_similarity(
        city=city,
        res=res,
        layers=layers,
        measure=measure,
        size=size,
    )
    
    hashed_similarities = (hashed_similarities + hashed_similarities.T)
        
    # 2) Flatten each to 1D with stack()
    hashed_stacked = hashed_similarities.stack()
    true_stacked = true_sim_matrix.stack()

    # 3) Combine into one DataFrame
    df_both = pd.DataFrame({
        "hashed": hashed_stacked,
        "true": true_stacked
    })
    
    # 4) Drop rows with any NaN
    df_both.dropna(inplace=True)
    
    # 5) Now compute correlation only over valid pairs
    corr = df_both["hashed"].corr(df_both["true"], method="pearson")
    return corr

# Varying layers
def compute_grid_corr_varying_layers(
    city: str,
    layers: list[int],
    true_sim_matrix,
    resolution: float = 1.6,
    size: int = 50,
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying layers"""

    results = []
    with Pool(parallel_jobs) as pool:
        for layer in layers: # Varying parameter
            corrs = pool.map(
                _fun_wrapper_corr,
                [
                    (city, resolution, layer, true_sim_matrix, measure, size)
                    for _ in range(parallel_jobs)
                ],
            )
            corr = np.average(np.array(corrs))
            std = np.std(np.array(corrs))
            results.append([corr, layer, std])

    return results

def plot_grid_corr_varying_layers(
     city: str,
    layers: list[int],
    true_sim_matrix,
    resolution: float = 1.6,
    size: int = 50,
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for layers for the grid hashes
    """

    results = compute_grid_corr_varying_layers(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, resolution = resolution, size=size, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    corre, layers, std = list(zip(*results))
    corre = np.array(corre)
    layers = np.array(layers)
    std = np.array(std)
    
    print(corre)


    ax1.plot(layers, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
    ax2.plot(layers, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
    
    ax2.text(
        0.37,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Grid) - {measure} True\nResolution: {resolution} km\nSize: {size}\nJobs: {parallel_jobs} ",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=12,
        color="grey",
    )
    
    
    ax1.set_xlabel("Number of layers", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid line", fontsize=18)
    ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
    ax2.set_ylabel("Standard deviation - Dashed line", fontsize=18)
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()



# Varying resolution
def compute_grid_corr_varying_resolution(
    city: str,
    layers: int,
    true_sim_matrix,
    resolution: list[float],
    size: int = 50,
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying resolutions"""

    pool = Pool()
    
    results = []
    
    for res in np.arange(*resolution): # Varying parameter
        corrs = pool.map(
            _fun_wrapper_corr,
            [
                (city, res, layers, true_sim_matrix, measure, size)
                for _ in range(parallel_jobs)
            ],
        )
        corr = np.average(np.array(corrs))
        std = np.std(np.array(corrs))
        results.append([corr, res, std])

    return results

def plot_grid_corr_varying_resolution(
    city: str,
    layers: int,
    true_sim_matrix,
    resolution: list[float],
    size: int = 50,
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution for the grid hashes
    """

    results = compute_grid_corr_varying_resolution(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, resolution = resolution, size=size, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    corre, res, std = list(zip(*results))
    corre = np.array(corre)
    res = np.array(res)
    std = np.array(std)
    
    ax1.plot(res, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
    ax2.plot(res, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
    # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    # ax1.legend(loc="lower left", ncols=3)
    ax2.text(
        0.37,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Grid) - {measure} True\nlayers: {layers}\nSize: {size}\nJobs: {parallel_jobs} ",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=12,
        color="grey",
    )
    
    
    ax1.set_xlabel("Grid resolution (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid line", fontsize=18)
    ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
    ax2.set_ylabel("Standard deviation - Dashed line", fontsize=18)
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()



import numpy as np
import pandas as pd
from multiprocessing import Pool

def compute_grid_corr_varying_resolution_and_layers(
    city: str,
    layers: list[int],  # Varying parameter
    true_sim_matrix,
    resolution: list[float],  # List of float values
    size: int = 50,
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computes correlation for resolution/layer configs and writes to CSV via DataFrame."""

    results = []
    df = pd.DataFrame(columns=["Resolution", "Layers", "Correlation"])

    with Pool(parallel_jobs) as pool:
        for layer in layers:
            result = []

            for res in resolution:
                print(f"L: {layer}", "{:.2f}".format(res), end="\r")

                corrs = pool.map(
                    _fun_wrapper_corr,
                    [
                        (city, res, layer, true_sim_matrix, measure, size)
                        for _ in range(parallel_jobs)
                    ],
                )

                corr = np.average(np.array(corrs))
                std = np.std(np.array(corrs))
                result.append([corr, res, std])

                df = pd.concat(
                    [df, pd.DataFrame([{
                        "Resolution": res,
                        "Layers": layer,
                        "Correlation": f"{corr:.3f}"
                    }])],
                    ignore_index=True
                )

                df.to_csv("correlation_results_resolution_layers.csv", index=False)

            results.append([result, layer])

    return results



def plot_grid_corr_varying_resolution_and_layers(
    city: str,
    layers: list[int], #Varying parameter
    true_sim_matrix,
    resolution: list[float], #Varying parameter
    size: int = 50,
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    results = compute_grid_corr_varying_resolution_and_layers(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, resolution = resolution, size=size, measure=measure, parallel_jobs=parallel_jobs
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
        color = COLOR_MAP[layer]

        ax1.plot(
            res,
            corre,
            c=color,
            label=f"{layer} layers",
            lw=2,
        )
        ax2.plot(res, std, c=color, alpha=0.3, ls="dashed")
        # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    ax1.legend(
        loc="center right",
        ncols=2,
        fontsize=16,
        labelspacing=0.2,
        borderpad=0.2,
        handlelength=1,
        handletextpad=0.5,
        borderaxespad=0.2,
        columnspacing=1,
    )
    ax2.text(
        0.01,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Grid) - {measure} True\nSize: {size}\nJobs: {parallel_jobs} ",
        ha="left",
        va="top",
        transform=ax2.transAxes,
        fontsize=11,
        color="black",
    )
    ax1.set_xlabel("Grid resolution (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax1.set_ylim([0, 1.0])
    ax2.set_ylabel("Standard deviation \- Dashed lines", fontsize=16)
    # Dynamic y-axis limits based on values
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    # ax2.set_ylim([0.0, 0.1])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()



# #varying size (TRENGS FIXING)
# def compute_d_corr_varying_size_bucketing(
#     city: str,
#     layers: int,
#     true_sim_matrix,
#     disks: int, 
#     diameter: float, 
#     sizes: list[int], #Varying parameter
#     bucketing_method: str = "original",
#     measure: str = "dtw",
#     parallel_jobs: int = 20,
# ):
#     """Computations for the visualisation with varying dataset sizes"""

#     pool = Pool()
    
#     results = []
    
#     for size in sizes: # Varying parameter
#         corrs = pool.map(
#             _fun_wrapper_corr_bucketing,
#             [
#                 (city, diameter, layers, disks, true_sim_matrix, measure, size, bucketing_method)
#                 for _ in range(parallel_jobs)
#             ],
#         )
#         corr = np.average(np.array(corrs))
#         std = np.std(np.array(corrs))
#         results.append([corr, size, std])

#     return results

# def plot_disk_corr_varying_size_bucketing(
#     city: str,
#     layers: int,
#     true_sim_matrix,
#     disks: int, 
#     diameter: float, 
#     sizes: list[int], #Varying parameter
#     bucketing_method: str = "original",
#     measure: str = "dtw",
#     parallel_jobs: int = 20,
# ):
#     """Visualises the 'optimal' values for resolution and layers for the disk hashes
#     """

#     results = compute_disk_corr_varying_size_bucketing(
#         city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter=diameter, disks=disks, sizes=sizes, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
#     )
    
#     fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
#     ax2 = ax1.twinx()
#     # fig.set_size_inches(10,8)
#     cmap = plt.get_cmap("gist_ncar")
#     N = len(results)

#     corre, size, std = list(zip(*results))
#     corre = np.array(corre)
#     size = np.array(size)
#     std = np.array(std)
    
#     ax1.plot(size, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
#     ax2.plot(size, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
#     # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

#     # Now styling the figure
#     # ax1.legend(loc="lower left", ncols=3)
#     ax2.text(
#         0.37,
#         0.99,
#         f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nDisks: {disks} km\nDiameter: {diameter}\nlayers: {layers}\nJobs: {parallel_jobs} ",
#         ha="right",
#         va="top",
#         transform=ax2.transAxes,
#         fontsize=12,
#         color="grey",
#     )
    
    
#     ax1.set_xlabel("Dataset size", fontsize=18)
#     ax1.set_ylabel("Pearson correlation coefficient - Solid line", fontsize=18)
#     ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
#     ax2.set_ylabel("Standard deviation - Dashed line", fontsize=18)
#     ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
#     ax1.tick_params(axis="both", which="major", labelsize=16)
#     ax2.tick_params(axis="both", which="major", labelsize=16)

#     plt.show()
