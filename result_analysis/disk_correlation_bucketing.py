"""
This file contains methods for finding an optimal/working number of disks and their diameter / layers

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
from computation.similarity import generate_disk_hash_similarity, generate_disk_hash_similarity_with_bucketing
from constants import COLOR_MAP, COLOR_MAP_DISKS

# Defining helper functions:
def mirrorDiagonal(M: np.ndarray) -> np.ndarray:
    """Flips and mirrors a two-dimenional np.array"""
    return M.values + np.rot90(np.fliplr(M.values))




# used for calculating correlation for one specific configuration 
def fun_wrapper_corr_bucketing(hashed_similarities, true_sim_matrix):
     
    # 1) Align columns and rows
    truesim_filtered = true_sim_matrix.loc[hashed_similarities.index, hashed_similarities.columns]

    # 2) Flatten each to 1D with stack()
    hashed_stacked = hashed_similarities.stack()
    true_stacked = truesim_filtered.stack()

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



# Brukes av alle funksjonene under
def _fun_wrapper_corr_bucketing(args):
    
    city, diameter, layers, disks, true_sim_matrix, measure, size, bucketing_method = args
    
    hashed_similarities, bucket_system, total_comparisons, total_skipped_comparisons, num_compared_trajectories = generate_disk_hash_similarity_with_bucketing(
        city=city,
        diameter=diameter,
        disks=disks,
        layers=layers,
        measure=measure,
        size=size,
        bucketing_method=bucketing_method
    )
    
    # 1) Align columns and rows
    truesim_filtered = true_sim_matrix.loc[hashed_similarities.index, hashed_similarities.columns]

    # 2) Flatten each to 1D with stack()
    hashed_stacked = hashed_similarities.stack()
    true_stacked = truesim_filtered.stack()

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
def compute_disk_corr_varying_layers_bucketing(
    city: str,
    layers: list[int],
    true_sim_matrix,
    diameter: float = 1.6,
    disks: int = 100,
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying layers"""

    pool = Pool()
    
    results = []
    
    for layer in layers: # Varying parameter
        corrs = pool.map(
            _fun_wrapper_corr_bucketing,
            [
                (city, diameter, layer, disks, true_sim_matrix, measure, size, bucketing_method)
                for _ in range(parallel_jobs)
            ],
        )
        corr = np.average(np.array(corrs))
        std = np.std(np.array(corrs))
        results.append([corr, layer, std])

    return results

def plot_disk_corr_varying_layers_bucketing(
    city: str,
    layers: list[int],
    true_sim_matrix,
    diameter: float = 1.6,
    disks: int = 100,
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution and layers for the disk hashes
    """

    results = compute_disk_corr_varying_layers_bucketing(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter= diameter, disks=disks, size=size, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    corre, layers, std = list(zip(*results))
    corre = np.array(corre)
    layers = np.array(layers)
    std = np.array(std)
    
    ax1.plot(layers, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
    ax2.plot(layers, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
    # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    # ax1.legend(loc="lower left", ncols=3)
    ax2.text(
        0.37,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nDiameter: {diameter} km\nSize: {size}\nDisks: {disks}\nJobs: {parallel_jobs} ",
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







# Varying #disks (LITT RAR, Skjønner ikke hvorfor den lister opp disker i desimaler)
def compute_disk_corr_varying_disks_bucketing(
    city: str,
    layers: int,
    true_sim_matrix,
    disks: list[int], #Varying parameter
    diameter: float = 1.6,
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying number of disks"""

    pool = Pool()
    
    results = []
    
    for num_disks in disks: # Varying parameter
        corrs = pool.map(
            _fun_wrapper_corr_bucketing,
            [
                (city, diameter, layers, num_disks, true_sim_matrix, measure, size, bucketing_method)
                for _ in range(parallel_jobs)
            ],
        )
        corr = np.average(np.array(corrs))
        std = np.std(np.array(corrs))
        results.append([corr, num_disks, std])

    return results

def plot_disk_corr_varying_disks_bucketing(
    city: str,
    layers: int,
    true_sim_matrix,
    disks: list[int], #Varying parameter
    diameter: float = 1.6,
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution and layers for the disk hashes
    """

    results = compute_disk_corr_varying_disks_bucketing(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter= diameter, disks=disks, size=size, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    corre, num_disks, std = list(zip(*results))
    corre = np.array(corre)
    num_disks = np.array(num_disks)
    std = np.array(std)
    
    ax1.plot(num_disks, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
    ax2.plot(num_disks, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
    # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    # ax1.legend(loc="lower left", ncols=3)
    ax2.text(
        0.37,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nDiameter: {diameter} km\nSize: {size}\nlayers: {layers}\nJobs: {parallel_jobs} ",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=12,
        color="grey",
    )
    
    
    ax1.set_xlabel("Number of disks", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid line", fontsize=18)
    ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
    ax2.set_ylabel("Standard deviation - Dashed line", fontsize=18)
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()







#Varying diameter
def compute_disk_corr_varying_diameter_bucketing(
    city: str,
    layers: int,
    true_sim_matrix,
    disks: int, 
    diameter: list[float], #Varying parameter
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying diameters"""

    pool = Pool()
    
    results = []
    
    ## USES NP.ARRANGE (like defining av range)
    for dia in np.arange(*diameter): # Varying parameter
        

        corrs = pool.map(
            _fun_wrapper_corr_bucketing,
            [
                (city, dia, layers, disks, true_sim_matrix, measure, size, bucketing_method)
                for _ in range(parallel_jobs)
            ],
        )
        corr = np.average(np.array(corrs))
        std = np.std(np.array(corrs))
        results.append([corr, dia, std])

    return results

def plot_disk_corr_varying_diameter_bucketing(
    city: str,
    layers: int,
    true_sim_matrix,
    disks: int, 
    diameter: list[float], #Varying parameter
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution and layers for the disk hashes
    """

    results = compute_disk_corr_varying_diameter_bucketing(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter=diameter, disks=disks, size=size, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    corre, dia, std = list(zip(*results))
    corre = np.array(corre)
    dia = np.array(dia)
    std = np.array(std)
    
    ax1.plot(dia, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
    ax2.plot(dia, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
    # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    # ax1.legend(loc="lower left", ncols=3)
    ax2.text(
        0.37,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nDisks: {disks} km\nSize: {size}\nlayers: {layers}\nJobs: {parallel_jobs} ",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=12,
        color="grey",
    )
    
    
    ax1.set_xlabel("Disk Diameter (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid line", fontsize=18)
    ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
    ax2.set_ylabel("Standard deviation - Dashed line", fontsize=18)
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()








#varying size (TRENGS FIXING)
def compute_disk_corr_varying_size_bucketing(
    city: str,
    layers: int,
    true_sim_matrix,
    disks: int, 
    diameter: float, 
    sizes: list[int], #Varying parameter
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying dataset sizes"""

    pool = Pool()
    
    results = []
    
    for size in sizes: # Varying parameter
        corrs = pool.map(
            _fun_wrapper_corr_bucketing,
            [
                (city, diameter, layers, disks, true_sim_matrix, measure, size, bucketing_method)
                for _ in range(parallel_jobs)
            ],
        )
        corr = np.average(np.array(corrs))
        std = np.std(np.array(corrs))
        results.append([corr, size, std])

    return results

def plot_disk_corr_varying_size_bucketing(
    city: str,
    layers: int,
    true_sim_matrix,
    disks: int, 
    diameter: float, 
    sizes: list[int], #Varying parameter
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Visualises the 'optimal' values for resolution and layers for the disk hashes
    """

    results = compute_disk_corr_varying_size_bucketing(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter=diameter, disks=disks, sizes=sizes, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    corre, size, std = list(zip(*results))
    corre = np.array(corre)
    size = np.array(size)
    std = np.array(std)
    
    ax1.plot(size, corre, c=cmap(float(1.3 - 1) / (1.2 * N)), lw=2)
    ax2.plot(size, std, c=cmap(float(1.3 - 1) / (1.2 * N)), ls="dashed")
    # plt.fill_between(res, np.array(corre)+np.array(std), np.array(corre)-np.array(std))

    # Now styling the figure
    # ax1.legend(loc="lower left", ncols=3)
    ax2.text(
        0.37,
        0.99,
        f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nDisks: {disks} km\nDiameter: {diameter}\nlayers: {layers}\nJobs: {parallel_jobs} ",
        ha="right",
        va="top",
        transform=ax2.transAxes,
        fontsize=12,
        color="grey",
    )
    
    
    ax1.set_xlabel("Dataset size", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid line", fontsize=18)
    ax1.set_ylim([ax1.get_ylim()[0] * 0.8, ax1.get_ylim()[1]])
    ax2.set_ylabel("Standard deviation - Dashed line", fontsize=18)
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()







# Varying diameter and layers
def compute_disk_corr_varying_diameter_and_layers_bucketing(
    city: str,
    layers: list[int], #Varying parameter
    true_sim_matrix,
    diameter: list[float], #Varying parameter
    disks: int = 100,
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying diameter and layers"""

    pool = Pool()
    
    results = []
    
    for layer in layers: # Varying parameter
        
        result = []
        
        for dia in np.arange(*diameter):
            print(f"L: {layer}", "{:.2f}".format(dia), end="\r")
            
            corrs = pool.map(
                _fun_wrapper_corr_bucketing,
                [
                    (city, dia, layer, disks, true_sim_matrix, measure, size, bucketing_method)
                    for _ in range(parallel_jobs)
                ],
            )
            
            corr = np.average(np.array(corrs))
            std = np.std(np.array(corrs))
            result.append([corr, dia, std])
        
        results.append([result, layer])

    return results

def plot_disk_corr_varying_diameter_and_layers_bucketing(
    city: str,
    layers: list[int], #Varying parameter
    true_sim_matrix,
    diameter: list[float], #Varying parameter
    disks: int = 100,
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    results = compute_disk_corr_varying_diameter_and_layers_bucketing(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter= diameter, disks=disks, size=size, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
    )
    
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    for layer_element in results:
        corrs, layer = layer_element

        corre, dia, std = list(zip(*corrs))
        corre = np.array(corre)
        dia = np.array(dia)
        std = np.array(std)
        color = COLOR_MAP[layer]

        ax1.plot(
            dia,
            corre,
            c=color,
            label=f"{layer} layers",
            lw=2,
        )
        ax2.plot(dia, std, c=color, alpha=0.3, ls="dashed")
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
        f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nSize: {size}\nDisks: {disks}\nJobs: {parallel_jobs} ",
        ha="left",
        va="top",
        transform=ax2.transAxes,
        fontsize=11,
        color="black",
    )
    ax1.set_xlabel("Disk diameter (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax1.set_ylim([0, 1.0])
    ax2.set_ylabel("Standard deviation \- Dashed lines", fontsize=16)
    # Dynamic y-axis limits based on values
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    # ax2.set_ylim([0.0, 0.1])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()



#varying disks and layers
def compute_disk_corr_varying_num_of_disks_and_layers_bucketing(
    city: str,
    layers: list[int], #Varying parameter
    true_sim_matrix,
    diameter: float,
    disks: list[int], #Varying parameter
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying number of disks and layers"""

    pool = Pool()
    
    results = []
    
    for layer in layers: # Varying parameter
        
        result = []
        
        for num_disks in disks:
            print(f"L: {layer}", "{:.2f}".format(num_disks), end="\r")
            
            corrs = pool.map(
                _fun_wrapper_corr_bucketing,
                [
                    (city, diameter, layer, num_disks, true_sim_matrix, measure, size, bucketing_method)
                    for _ in range(parallel_jobs)
                ],
            )
            
            corr = np.average(np.array(corrs))
            std = np.std(np.array(corrs))
            result.append([corr, num_disks, std])
        
        results.append([result, layer])

    return results

def plot_disk_corr_varying_num_of_disks_and_layers_bucketing(
    city: str,
    layers: list[int], #Varying parameter
    true_sim_matrix,
    diameter: float,
    disks: list[int], #Varying parameter
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    results = compute_disk_corr_varying_num_of_disks_and_layers_bucketing(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter=diameter, disks=disks, size=size, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    for layer_element in results:
        corrs, layer = layer_element

        corre, num_disks, std = list(zip(*corrs))
        corre = np.array(corre)
        num_disks = np.array(num_disks)
        std = np.array(std)
        color = COLOR_MAP[layer]

        ax1.plot(
            num_disks,
            corre,
            c=color,
            label=f"{layer} layers",
            lw=2,
        )
        ax2.plot(num_disks, std, c=color, alpha=0.3, ls="dashed")
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
        f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nSize: {size}\nDiameter: {diameter}\nJobs: {parallel_jobs} ",
        ha="left",
        va="top",
        transform=ax2.transAxes,
        fontsize=11,
        color="black",
    )
    ax1.set_xlabel("Number of disks", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax1.set_ylim([0, 1.0])
    ax2.set_ylabel("Standard deviation \- Dashed lines", fontsize=16)
    # Dynamic y-axis limits based on values
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    # ax2.set_ylim([0.0, 0.1])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()



#varying disks and diameter
def compute_disk_corr_varying_num_of_disks_and_diameter_bucketing(
    city: str,
    layers: int, 
    true_sim_matrix,
    diameter: list[float], #Varying parameter
    disks: list[int], #Varying parameter
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    """Computations for the visualisation with varying number of disks and diameter"""

    pool = Pool()
    
    results = []
    
    for num_disks in disks: # Varying parameter
        
        result = []
        
        for dia in np.arange(*diameter):
            print(f"D: {num_disks}", "{:.2f}".format(dia), end="\r")
            
            corrs = pool.map(
                _fun_wrapper_corr_bucketing,
                [
                    (city, dia, layers, num_disks, true_sim_matrix, measure, size, bucketing_method)
                    for _ in range(parallel_jobs)
                ],
            )
            
            corr = np.average(np.array(corrs))
            std = np.std(np.array(corrs))
            result.append([corr, dia, std])
        
        results.append([result, num_disks])

    return results

def plot_disk_corr_varying_num_of_disks_and_diameter_bucketing(
    city: str,
    layers: list[int], #Varying parameter
    true_sim_matrix,
    diameter: int,
    disks: list[int], #Varying parameter
    size: int = 50,
    bucketing_method: str = "original",
    measure: str = "dtw",
    parallel_jobs: int = 20,
):
    results = compute_disk_corr_varying_num_of_disks_and_diameter_bucketing(
        city=city, layers=layers, true_sim_matrix=true_sim_matrix, diameter=diameter, disks=disks, size=size, bucketing_method=bucketing_method, measure=measure, parallel_jobs=parallel_jobs
    )
    
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = ax1.twinx()
    # fig.set_size_inches(10,8)
    cmap = plt.get_cmap("gist_ncar")
    N = len(results)

    for num_disk_element in results:
        corrs, num_disks = num_disk_element

        corre, dia, std = list(zip(*corrs))
        corre = np.array(corre)
        dia = np.array(dia)
        std = np.array(std)
        color = COLOR_MAP_DISKS[num_disks]

        ax1.plot(
            dia,
            corre,
            c=color,
            label=f"{num_disks} Disks",
            lw=2,
        )
        ax2.plot(dia, std, c=color, alpha=0.3, ls="dashed")
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
        f"{city.capitalize()}: {measure.upper()} (Disk) - {measure} True\nSize: {size}\nLayers: {layers}\nJobs: {parallel_jobs} ",
        ha="left",
        va="top",
        transform=ax2.transAxes,
        fontsize=11,
        color="black",
    )
    ax1.set_xlabel("Disk Diameter (km)", fontsize=18)
    ax1.set_ylabel("Pearson correlation coefficient - Solid lines", fontsize=18)
    ax1.set_ylim([0, 1.0])
    ax2.set_ylabel("Standard deviation \- Dashed lines", fontsize=16)
    # Dynamic y-axis limits based on values
    ax2.set_ylim([0, ax2.get_ylim()[1] * 2])
    # ax2.set_ylim([0.0, 0.1])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)

    plt.show()