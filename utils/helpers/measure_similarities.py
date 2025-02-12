from multiprocessing import Pool
import os, sys
import time
import timeit as ti
import pandas as pd

from schemes.lsh_bucketing import place_hashes_into_buckets, place_hashes_into_buckets_individual

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

from computation import similarity
from utils.similarity_measures import dtw, frechet, hashed_dtw, hashed_frechet
from computation.similarity import (
    measure_hashed_cy_bucketing,
    transform_np_numerical_disk_hashes_to_non_np,
)
from utils.helpers import file_handler as fh
from utils.helpers import metafile_handler as mfh
from computation.similarity import _constructDisk, _constructGrid


def get_dataset_path(city: str) -> str:
    return f"../../../dataset/{city}/output/"

print(get_dataset_path("porto"))



sim = {
    #True cython
    "true_dtw_cy": dtw.measure_cy_dtw,
    "true_frechet_cy": frechet.measure_cy_frechet,
    
    #Schemes cython
    "disk_dtw_cy": hashed_dtw.measure_hashed_cy_dtw,
    "disk_frechet_cy": hashed_frechet.measure_hashed_cy_frechet,
    "grid_dtw_cy": hashed_dtw.measure_hashed_cy_dtw,
    "grid_frechet_cy": hashed_frechet.measure_hashed_cy_frechet,
    
    #Python
    "true_dtw_py": dtw.measure_py_dtw,
    "true_frechet_py": frechet.measure_py_frechet,
    "disk_dtw_py": hashed_dtw.measure_hashed_py_dtw,
    "grid_dtw_py": hashed_dtw.measure_hashed_py_dtw,
}


def measure_true_similarities(
    measure: str, data_folder: str, meta_file: str, parallel_jobs: int = 10
):
    """Common method for measuring the efficiency of the similarity algorithms"""
    files = mfh.read_meta_file(meta_file)
    trajectories = fh.load_trajectory_files(files, data_folder)

    with Pool() as pool:
        result = pool.map(
            sim[measure], [[trajectories, 1, 1] for _ in range(parallel_jobs)]
        )
    return result


def compute_true_similarity_runtimes(
    measure: str,
    city: str,
    parallel_jobs: int = 10,
    data_start_size: int = 100,
    data_end_size: int = 1000,
    data_step_size: int = 100,
    iterations: int = 3
):
    
    #Path to data folder
    data_folder = get_dataset_path(city)
    data_sets = range(data_start_size, data_end_size + 1, data_step_size)
    print("data_end_size", data_end_size)
    print("data sets", data_sets)
    
    #Output folder
    output_folder = f"../../../results_true/runtimes/{city}/"
    #Filename    
    file_name = f"similarity_runtimes_{measure}_start-{data_start_size}_end-{data_end_size}_step-{data_step_size}.csv"
    
    # Initialize a list to hold the DataFrames from each iteration
    dfs_iterations = []

    # Initialize an empty DataFrame to accumulate execution times
    df_accumulated = pd.DataFrame()
    print(f"Computing {measure} for {city} with {parallel_jobs} jobs")

    for iteration in range(iterations):  # Loop through each iteration
        print(f"Iteration {iteration+1}/{iterations}")
        df = pd.DataFrame(
            index=[f"run_{x+1}" for x in range(parallel_jobs)],
            columns=[x for x in data_sets],
        )

        index = 1
        for size in data_sets:
            print(f"Computing size {size}, set {index}/{len(data_sets)}", end="\r")
            meta_file = f"{data_folder}META-{size}.txt"
            print(meta_file)
            execution_times = measure_true_similarities(
                measure=measure,
                data_folder=data_folder,
                meta_file=meta_file,
                parallel_jobs=parallel_jobs,
            )
            df[size] = [element[0] for element in execution_times]
            index += 1
        # Append the DataFrame of this iteration to the list
        dfs_iterations.append(df)
        print(dfs_iterations)

        # Accumulate the results from this iteration
        if df_accumulated.empty:
            df_accumulated = df.copy()
        else:
            df_accumulated += df

    # Calculate the average execution times over all iterations
    df_average = pd.concat(dfs_iterations).groupby(level=0).mean()

    # Print the runtimes of each iteration for comparison
    for i, df_iteration in enumerate(dfs_iterations, 1):
        print(f"\nRuntimes - Iteration {i}:")
        print(df_iteration)

    print("\nAverage Runtimes:")
    print(df_average)
    # Optionally, save the average execution times to a CSV file
    df_average.to_csv(os.path.join(output_folder, file_name))


def measure_hashed_similarities(args):
    hashes, measure = args
    elapsed_time = time.timeit(
        lambda: sim[measure](hashes), number=1, timer=time.process_time
    )
    return elapsed_time


def compute_disk_hashes(city: str, diameter: float, layers: int, disks: int, size: int):
    disk = similarity._constructDisk(city, diameter, layers, disks, size)
    return disk.compute_dataset_hashes_with_KD_tree_numerical()
    


def compute_grid_hashes(city: str, res: float, layers: int, size: int):
    grid = _constructGrid(city, res, layers, size)
    return grid.compute_dataset_hashes()


def compute_hashed_similarity_runtimes(
    measure: str,
    city: str,
    layers: int = 4,
    res: float = 0.5,
    diameter: float = 0.5,
    disks: int = 100,
    parallel_jobs: int = 10,
    data_start_size: int = 100,
    data_end_size: int = 1000,
    data_step_size: int = 100,
    iterations: int = 3,  # New parameter for specifying the number of iterations
):
    """
    Function that measures runtime for:
        * Hashing (either with grid or disk)
        * Similarity computation: Time it takes to compute the similarity value between all trajectory hashes

        Raises:
            ValueError: 
    """
    use_fixed_dataset_size = False
    resolutions = [
        round(0.2 * i, 2) for i in range(1, 11)
    ]  # From 0.2 to 2.0 in steps of 0.2
    size = 1000  # Fixed dataset size

    # Set this to true if you want to save the hash generation times to a file
    save_hash_generation = False

    data_sets = range(data_start_size, data_end_size + 1, data_step_size)
    output_folder = "../benchmarks/similarities_runtimes/"
    scheme = "grid" if "grid" in measure else "disk"
    # Adjust file_name generation based on measure
    if measure in ["disk_dtw_cy", "disk_frechet_cy", "disk_dtw_py"]:
        if use_fixed_dataset_size:
            file_name = f"{scheme}/{city}/similarity_runtimes_{measure}_diameter-{diameter}_disks-{disks}_{city}_size-{size}.csv"
        else:
            file_name = f"{scheme}/{city}/similarity_runtimes_{measure}_layers-{layers}_diameter-{diameter}_disks-{disks}_{city}_start-{data_start_size}_end-{data_end_size}_step-{data_step_size}.csv"
    elif measure in ["grid_dtw_cy", "grid_frechet_cy", "grid_dtw_py"]:
        if use_fixed_dataset_size:
            file_name = f"{scheme}/{city}/similarity_runtimes_{measure}_layers-{layers}_{city}_size-{size}.csv"
        else:
            file_name = f"{scheme}/{city}/similarity_runtimes_{measure}_layers-{layers}_res-{res}_{city}_start-{data_start_size}_end-{data_end_size}_step-{data_step_size}.csv"

    # Initialize a list to hold the DataFrames from each iteration
    dfs_iterations = []
    hash_generation_times = {size: [] for size in data_sets}

    # Initialize an empty DataFrame to accumulate execution times
    df_accumulated = pd.DataFrame()

    for iteration in range(iterations):  # Loop through each iteration
        print(f"Iteration {iteration+1}/{iterations}")
        if use_fixed_dataset_size:
            if measure in ["grid_dtw_cy", "grid_frechet_cy"]:
                df = pd.DataFrame(
                    index=[f"run_{x+1}" for x in range(parallel_jobs)],
                    columns=[x for x in resolutions],
                )
            elif measure in ["disk_dtw_cy", "disk_frechet_cy"]:
                df = pd.DataFrame(
                    index=[f"run_{x+1}" for x in range(parallel_jobs)],
                    columns=[x for x in range(1, 6)],
                )
        else:
            df = pd.DataFrame(
                index=[f"run_{x+1}" for x in range(parallel_jobs)],
                columns=[x for x in data_sets],
            )

        print(
            f"Computing {measure} for {city} with {parallel_jobs} jobs - Iteration {iteration+1}/{iterations}"
        )
        index = 1

        # NOTE - Standard implementation
        if not use_fixed_dataset_size:
            for size in data_sets:
                elapsed_time_for_hash_generation = 0
                print(f"Computing size {size}, set {index}/{len(data_sets)}", end="\r")
                with Pool(parallel_jobs) as pool:
                    start_time = time.perf_counter()
                    if measure in ["disk_dtw_cy", "disk_frechet_cy", "disk_dtw_py"]:
                        hashes = compute_disk_hashes(
                            city=city,
                            diameter=diameter,
                            layers=layers,
                            disks=disks,
                            size=size,
                        )
                    elif measure in ["grid_dtw_cy", "grid_frechet_cy", "grid_dtw_py"]:
                        hashes = compute_grid_hashes(
                            city=city, res=res, layers=layers, size=size
                        )
                    else:
                        raise ValueError("Invalid measure")
                    end_time = time.perf_counter()
                    elapsed_time_for_hash_generation += end_time - start_time

                    hash_generation_times[size].append(elapsed_time_for_hash_generation)

                    execution_times = pool.map(
                        sim[measure], [hashes for _ in range(parallel_jobs)]
                    )

                df[size] = [element[0] for element in execution_times]
                index += 1

        # NOTE - Fixed datasize with increasing resolution
        else:
            if measure in ["grid_dtw_cy", "grid_frechet_cy"]:
                for res in resolutions:
                    print(f"Resolution {res}", end="\r")
                    with Pool(parallel_jobs) as pool:
                        hashes = compute_grid_hashes(
                            city=city, res=res, layers=layers, size=size
                        )
                        execution_times = pool.map(
                            sim[measure], [hashes for _ in range(parallel_jobs)]
                        )
                    df[res] = [element[0] for element in execution_times]
            elif measure in ["disk_dtw_cy", "disk_frechet_cy"]:
                for layer in range(1, 6):
                    print(f"Layer {layer}", end="\r")
                    with Pool(parallel_jobs) as pool:
                        hashes = compute_disk_hashes(
                            city=city,
                            diameter=diameter,
                            layers=layer,
                            disks=disks,
                            size=size,
                        )
                        execution_times = pool.map(
                            sim[measure], [hashes for _ in range(parallel_jobs)]
                        )
                    df[layer] = [element[0] for element in execution_times]
            else:
                raise ValueError("Invalid measure")

        # Append the DataFrame of this iteration to the list
        dfs_iterations.append(df)
        # Accumulate the results from this iteration
        if df_accumulated.empty:
            df_accumulated = df.copy()
        else:
            df_accumulated += df

    # Calculate the average execution times over all iterations
    # df_average = df_accumulated / iterations

    # Calculate the average execution times over all iterations
    df_average = pd.concat(dfs_iterations).groupby(level=0).mean()

    # Print the runtimes of each iteration for comparison
    for i, df_iteration in enumerate(dfs_iterations, 1):
        print(f"\nRuntimes - Iteration {i}:")
        print(df_iteration)

    if save_hash_generation:
        # Print the average hash generation time for each dataset size
        print("\nAverage Hash Generation Times:")
        for size in data_sets:
            average_time = sum(hash_generation_times[size]) / len(
                hash_generation_times[size]
            )
            print(f"Size {size}: {average_time:.4f} seconds")

    print("\nAverage Runtimes:")
    print(df_average)

    df_average.to_csv(os.path.join(output_folder, file_name))

    if save_hash_generation:
        hash_generation_file_name = file_name.replace(
            "similarity_runtimes", "hash_generation"
        )
        hash_generation_df = pd.DataFrame(
            {
                "Dataset Size": list(hash_generation_times.keys()),
                "Average Hash Generation Time (seconds)": [
                    sum(times) / len(times) for times in hash_generation_times.values()
                ],
            }
        )
    if save_hash_generation:
        hash_generation_df.to_csv(
            os.path.join(output_folder, hash_generation_file_name), index=False
        )


def compute_grid_similarity_runtimes(
    measure: str,
    city: str,
    res: float,
    layers: int,
    parallel_jobs: int = 10,
    data_start_size: int = 100,
    data_end_size: int = 1000,
    data_step_size: int = 100,
    iterations: int = 3,
):
    compute_hashed_similarity_runtimes(
        measure=measure,
        city=city,
        res=res,
        layers=layers,
        parallel_jobs=parallel_jobs,
        data_start_size=data_start_size,
        data_end_size=data_end_size,
        data_step_size=data_step_size,
        iterations=iterations,
    )


def compute_disk_similarity_runtimes(
    measure: str,
    city: str,
    diameter: float,
    layers: int,
    disks: int,
    parallel_jobs: int = 10,
    data_start_size: int = 100,
    data_end_size: int = 1000,
    data_step_size: int = 100,
    iterations: int = 3,
):
    compute_hashed_similarity_runtimes(
        measure=measure,
        city=city,
        diameter=diameter,
        layers=layers,
        disks=disks,
        parallel_jobs=parallel_jobs,
        data_start_size=data_start_size,
        data_end_size=data_end_size,
        data_step_size=data_step_size,
        iterations=iterations,
    )




################### NEW CODE - BUCKETING ####################

def compute_hashed_similarity_runtimes_with_bucketing(
    measure: str,
    city: str,
    layers: int = 4,
    res: float = 0.5,
    diameter: float = 0.5,
    disks: int = 100,
    parallel_jobs: int = 10,
    data_size: int = 100,
    iterations: int = 3,
):
    
    """
    Function that measures runtime for:
    * Hashing (either with grid or disk).
    * Bucket distribution: Time it takes to place the hashes into buckets.
    * Similarity computation over all buckets: Time it takes to compute the similarity values between trajectories in the same bucket for all buckets.

    Raises:
        ValueError: If measure not exists
    """

    #File handling
    scheme = "grid" if "grid" in measure else "disk"
    output_folder = f"../../../results_hashed/runtimes/{scheme}/{city}/"
    file_name  = filename_generator(measure=measure, city=city, layers=layers, res=res, diameter=diameter, disks=disks, data_size=data_size)
    
    # Initialize a list to hold the DataFrames from each iteration
    dfs_iterations = []
    
    hash_generation_times = []
    bucket_distribution_times = []

    # Initialize an empty DataFrame to accumulate execution times
    df_accumulated = pd.DataFrame()


    # Building the dataframes
    for iteration in range(iterations):  # Loop through each iteration
        print(f"Iteration {iteration+1}/{iterations}")
        
        column_name = f"data size:{data_size}"
        
        df = pd.DataFrame(
                index=[f"similarity computation run_{x+1}" for x in range(parallel_jobs)],
                columns=[column_name],
        )

        print(
            f"Computing {measure} for {city} with {parallel_jobs} jobs - Iteration {iteration+1}/{iterations}"
        )
        index = 1

        #Initializes times for hashing and bucketing
        elapsed_time_for_hash_generation = 0
        elapsed_time_for_bucket_distribution = 0
        
        print(f"Computing size {data_size}", end="\r")
        with Pool(parallel_jobs) as pool:
            
            #Hashing start
            start_time_hashing = time.perf_counter()            
            if measure in ["disk_dtw_cy", "disk_frechet_cy", "disk_dtw_py" ]: # -> DISK
                hashes = compute_disk_hashes(
                    city=city,
                    diameter=diameter,
                    layers=layers,
                    disks=disks,
                    size=data_size,
                )
            elif measure in ["grid_dtw_cy", "grid_frechet_cy", "grid_dtw_py"]:
                hashes = compute_grid_hashes(
                    city=city, res=res, layers=layers, size=data_size
                )
            else:
                raise ValueError("Invalid measure")
            end_time_hashing = time.perf_counter()
            #Hashing end
            
            elapsed_time_for_hash_generation += end_time_hashing - start_time_hashing
            hash_generation_times.append(elapsed_time_for_hash_generation)


            #Bucketing start
            start_time_bucketing = time.perf_counter()
            bucket_system = place_hashes_into_buckets_individual(hashes)
            end_time_bucketing = time.perf_counter()
            #Bucketing end
            
            elapsed_time_for_bucket_distribution += end_time_bucketing - start_time_bucketing
            bucket_distribution_times.append(elapsed_time_for_bucket_distribution)


           #Similarity computation start
            if "dtw" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing, [(hashes, scheme, "dtw", bucket_system, False) for _ in range(parallel_jobs)]
                )
            
            elif "frechet" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing, [(hashes, scheme, "frechet", bucket_system, False) for _ in range(parallel_jobs)]
                )
            #Similarity computation end
            
        df[column_name] = [element[0] for element in execution_times]
        index += 1
            
        # Append the DataFrame of this iteration to the list
        dfs_iterations.append(df)
        
        # Accumulate the results from this iteration
        if df_accumulated.empty:
            df_accumulated = df.copy()
        else:
            df_accumulated += df

    # Calculate the average execution times over all iterations
    df_final = pd.concat(dfs_iterations).groupby(level=0).mean()
    
    # Add hash generation time to the final DataFrame
    df_final.loc["Average hash generation time (Seconds)", "data size:50"] = sum(hash_generation_times) / len(hash_generation_times)

    # Add bucket distribution time to the final DataFrame
    df_final.loc["Average bucket distribution time (Seconds)", "data size:50"] = sum(bucket_distribution_times) / len(bucket_distribution_times)

    # Print the runtimes of each iteration for comparison
    for i, df_iteration in enumerate(dfs_iterations, 1):
        print(f"\nRuntimes - Iteration {i}:")
        print(df_iteration)

    # Save similarity computation runtimes to file
    df_final.to_csv(os.path.join(output_folder, file_name))
 

def filename_generator(
    measure: str,
    city: str,
    layers: int = 4,
    res: float = 0.5,
    diameter: float = 0.5,
    disks: int = 100,
    data_size: int = 100):
    
    file_name= ""
    
    scheme = "grid" if "grid" in measure else "disk"
    # Adjust file_name generation based on measure
    if measure in ["disk_dtw_cy", "disk_frechet_cy", "disk_dtw_py"]:
        file_name = f"{city}_similarity_runtimes_{measure}_diameter_{diameter}_layers_{layers}_disks_{disks}_trajectories_{data_size}.csv"
        
    elif measure in ["grid_dtw_cy", "grid_frechet_cy", "grid_dtw_py"]:
        file_name = f"{city}_similarity_runtimes_{measure}_resolution_{res}_layers-{layers}_trajectories_{data_size}.csv"
            
    return file_name