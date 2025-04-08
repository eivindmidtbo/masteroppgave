from multiprocessing import Pool
import os, sys
import time
import timeit as ti
import pandas as pd

from schemes.lsh_bucketing import *

# def find_project_root(target_folder="masteroppgave"):
#     """Find the absolute path of a folder by searching upward."""
#     currentdir = os.path.abspath("__file__")  # Get absolute script path
#     while True:
#         if os.path.basename(currentdir) == target_folder:
#             return currentdir  # Found the target folder
#         parentdir = os.path.dirname(currentdir)
#         if parentdir == currentdir:  # Stop at filesystem root
#             return None
#         currentdir = parentdir  # Move one level up

# project_root = find_project_root("masteroppgave")

# if project_root:
#     sys.path.append(project_root)
#     print(f"Project root found: {project_root}")
# else:
#     raise RuntimeError("Could not find 'masteroppgave' directory")

from computation import similarity
from utils.similarity_measures import dtw, frechet, hashed_dtw, hashed_frechet
from computation.similarity import (
    measure_hashed_cy_bucketing,
    measure_hashed_cy_bucketing_with_true_sim,
    transform_np_numerical_disk_hashes_to_non_np,
)
from utils.helpers import file_handler as fh
from utils.helpers import metafile_handler as mfh
from computation.similarity import _constructDisk, _constructGrid


def get_dataset_path(city: str) -> str:
    return f"../../../dataset/{city}/output/"

# Blir brukt av alt
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
    data_size: int = 100,
    iterations: int = 3,
):
    """
    Function that measures runtime for:
        * Hashing (either with grid or disk)
        * Similarity computation: Time it takes to compute the similarity value between all trajectory hashes

        Raises:
            ValueError: 
    """
 
    scheme = "grid" if "grid" in measure else "disk"
    
    hash_generation_times = []
    all_similarity_runtimes = []
    
    with Pool(parallel_jobs) as pool:
        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")

            # --- Parallel Hashing ---
            hash_results = pool.starmap(
                compute_hash_parallel, 
                [(measure, city, data_size, diameter, layers, disks, res) for _ in range(parallel_jobs)]
            )
            
            # Extract hashes and compute average hashing time
            hashes_list, hashing_times = zip(*hash_results)  # Unpack results
            avg_hashing_time = sum(hashing_times) / parallel_jobs  # Compute average
            hash_generation_times.append(avg_hashing_time)
            
            
            # --- Parallel Similarity Computation ---
            execution_times = pool.map(
                sim[measure], [hashes_list[i] for i in range(parallel_jobs)]
            )
            
            # Collect all runtime values
            avg_sim_comp_time = sum([element[0] for element in execution_times]) / parallel_jobs  # Compute average for this iteration
            all_similarity_runtimes.append(avg_sim_comp_time) #Add to list       

    # Compute overall averages
    overall_avg_sim_comp_time = format(sum(all_similarity_runtimes) / len(all_similarity_runtimes), ".3f")
    avg_hash_time = format(sum(hash_generation_times) / len(hash_generation_times), ".3f")

    #Total time
    total_time = format(float(overall_avg_sim_comp_time) + float(avg_hash_time), ".3f")

    # Create a single-row DataFrame
    df_final = pd.DataFrame(
        {
            "Size": [data_size],
            "Average Similarity Computation Time (Seconds)": [overall_avg_sim_comp_time],
            "Average Hash Generation Time (Seconds)": [avg_hash_time],
            "Total time (Seconds)": [total_time],
        }
    )

    return df_final

################### NEW CODE - BUCKETING ####################

BUCKETING_FUNCTION_MAP = {
    "original": place_hashes_into_buckets_original,
    "loose": place_hashes_into_buckets_loose
}

def compute_hash_parallel(measure, city, data_size, diameter, layers, disks, res):
        """Function to compute hashes in parallel"""
        start_time = time.perf_counter()
        if measure in ["disk_dtw_cy", "disk_frechet_cy", "disk_dtw_py"]:  # -> DISK
            hashes = compute_disk_hashes(city=city, diameter=diameter, layers=layers, disks=disks, size=data_size)
        elif measure in ["grid_dtw_cy", "grid_frechet_cy", "grid_dtw_py"]: # -> GRID
            hashes = compute_grid_hashes(city=city, res=res, layers=layers, size=data_size)
        else:
            raise ValueError("Invalid measure")
        end_time = time.perf_counter()
        return hashes, end_time - start_time  # Return both the hash and the time taken


def timed_bucketing(h, bucketing_method):
                """Wrap the bucketing function to measure time per call."""
                start = time.perf_counter()
                bucket_system = BUCKETING_FUNCTION_MAP[bucketing_method](h)
                end = time.perf_counter()
                return bucket_system, (end - start)  # Return both result and time

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
    bucketing_method: str = "original"
):
    """
    Function that measures runtime for:
    * Hashing (either with grid or disk).
    * Bucket distribution: Time it takes to place the hashes into buckets.
    * Similarity computation over all buckets: Time it takes to compute the similarity values between trajectories in the same bucket for all buckets.

    Raises:
        ValueError: If measure not exists
    """

    # File handling
    scheme = "grid" if "grid" in measure else "disk"

    # Initialize lists to collect runtime data
    hash_generation_times = []
    bucket_distribution_times = []
    all_similarity_runtimes = []


    with Pool(parallel_jobs) as pool:
        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")

            # --- Parallel Hashing ---
            hash_results = pool.starmap(
                compute_hash_parallel, 
                [(measure, city, data_size, diameter, layers, disks, res) for _ in range(parallel_jobs)]
            )

            # Extract hashes and compute average hashing time
            hashes_list, hashing_times = zip(*hash_results)  # Unpack results
            avg_hashing_time = sum(hashing_times) / parallel_jobs  # Compute average
            hash_generation_times.append(avg_hashing_time)

            # --- Parallel Bucketing ---
            bucket_results = pool.starmap(
                timed_bucketing, [(h, bucketing_method) for h in hashes_list]
            )

            # Extract bucket systems and compute average bucketing time
            bucket_systems, individual_bucketing_times = zip(*bucket_results)
            avg_bucket_time = sum(individual_bucketing_times) / parallel_jobs  # Compute average
            bucket_distribution_times.append(avg_bucket_time)

            # --- Parallel Similarity Computation ---
            if "dtw" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing, 
                    [(hashes_list[i], scheme, "dtw", bucket_systems[i], False) for i in range(parallel_jobs)]
                )
            elif "frechet" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing, 
                    [(hashes_list[i], scheme, "frechet", bucket_systems[i], False) for i in range(parallel_jobs)]
                )

            # Collect all runtime values
            avg_sim_comp_time = sum([element[0] for element in execution_times]) / parallel_jobs  # Compute average for this iteration
            all_similarity_runtimes.append(avg_sim_comp_time) #Add to list

    # Compute overall averages
    overall_avg_sim_comp_time = format(sum(all_similarity_runtimes) / len(all_similarity_runtimes), ".3f")
    avg_hash_time = format(sum(hash_generation_times) / len(hash_generation_times), ".3f")
    avg_bucket_time = format(sum(bucket_distribution_times) / len(bucket_distribution_times), ".3f")

    #Total time
    total_time = format(float(overall_avg_sim_comp_time) + float(avg_hash_time) + float(avg_bucket_time), ".3f")

    # Create a single-row DataFrame
    df_final = pd.DataFrame(
        {
            "Size": [data_size],
            "Average Similarity Computation Time (Seconds)": [overall_avg_sim_comp_time],
            "Average Hash Generation Time (Seconds)": [avg_hash_time],
            "Average Bucket Distribution Time (Seconds)": [avg_bucket_time],
            "Total time (Seconds)": [total_time],
        }
    )

    return df_final


def compute_hashed_similarity_runtimes_with_bucketing_with_true_sim(
    measure: str,
    city: str,
    layers: int = 4,
    res: float = 0.5,
    diameter: float = 0.5,
    disks: int = 100,
    parallel_jobs: int = 8,
    data_size: int = 100,
    iterations: int = 3,
    bucketing_method: str = "original"
):
    """
    Function that measures runtime for:
    * Hashing (either with grid or disk).
    * Bucket distribution: Time it takes to place the hashes into buckets.
    * Similarity computation over all buckets: Time it takes to compute the similarity values between trajectories in the same bucket for all buckets.

    Raises:
        ValueError: If measure not exists
    """

    # File handling
    scheme = "grid" if "grid" in measure else "disk"
    
    hash_generation_times = []
    bucket_distribution_times = []
    all_similarity_runtimes=[]

    
    # --- Main Loop ---
    with Pool(parallel_jobs) as pool:
        for iteration in range(iterations):
            print(f"Computing {measure} for {city} with {parallel_jobs} jobs - Iteration {iteration+1}/{iterations}")

            # --- Parallel Hashing ---
            hash_results = pool.starmap(
                compute_hash_parallel, 
                [(measure, city, data_size, diameter, layers, disks, res) for _ in range(parallel_jobs)]
            )

            # Extract hashes and compute total hashing time
            hashes_list, hashing_times = zip(*hash_results)  # Unpack results
            avg_hashing_time = sum(hashing_times) / parallel_jobs  # Compute the average time
            hash_generation_times.append(avg_hashing_time)

            # --- Parallel Bucketing --- 
            bucket_results = pool.starmap(
                timed_bucketing, [(h, bucketing_method) for h in hashes_list]
            )

            # Extract bucket systems and times
            bucket_systems, individual_bucketing_times = zip(*bucket_results)
            avg_bucket_time = sum(individual_bucketing_times) / parallel_jobs  # Compute average
            bucket_distribution_times.append(avg_bucket_time)

            # Load true trajectory coordinates
            true_coordinates = fh.load_trajectories_from_meta_file(data_size, f"../../../dataset/{city}/output/")

            # --- Parallel Similarity Computation ---
            if "dtw" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing_with_true_sim, 
                    [(true_coordinates, scheme, "dtw", bucket_systems[i], False) for i in range(parallel_jobs)]
                )
            elif "frechet" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing_with_true_sim, 
                    [(true_coordinates, scheme, "frechet", bucket_systems[i], False) for i in range(parallel_jobs)]
                )
            
            # Collect all runtime values
            avg_sim_comp_time = sum([element[0] for element in execution_times]) / parallel_jobs  # Compute average for this iteration
            all_similarity_runtimes.append(avg_sim_comp_time) #Add to list

    # Compute overall averages
    overall_avg_sim_comp_time = format(sum(all_similarity_runtimes) / len(all_similarity_runtimes), ".3f")
    avg_hash_time = format(sum(hash_generation_times) / len(hash_generation_times), ".3f")
    avg_bucket_time = format(sum(bucket_distribution_times) / len(bucket_distribution_times), ".3f")

    # Total time
    total_time = format(float(overall_avg_sim_comp_time) + float(avg_hash_time) + float(avg_bucket_time), ".3f")

    # Create a single-row DataFrame
    df_final = pd.DataFrame(
        {
            "Size": [data_size],
            "Average Similarity Computation Time (Seconds)": [overall_avg_sim_comp_time],
            "Average Hash Generation Time (Seconds)": [avg_hash_time],
            "Average Bucket Distribution Time (Seconds)": [avg_bucket_time],
            "Total time (Seconds)": [total_time],
        }
    )
    
    return df_final


def compute_hashed_similarity_runtimes_with_bucketing_hybrid(
    measure: str,
    city: str,
    layers_bucketing: int = 4,
    layers_compression: int = 4,
    res_bucketing: float = 0.5,
    res_compression: float = 0.5,
    diameter_bucketing: float = 0.5,
    diameter_compression: float = 0.5,
    disks_bucketing: int = 50,
    disks_compression: int = 50,
    parallel_jobs: int = 10,
    data_size: int = 100,
    iterations: int = 3,
    bucketing_method: str = "original"
): 
    """
    Function that measures runtime for:
    * Hashing (bucketing hash & compression hash) in parallel.
    * Bucket distribution: Time it takes to place the hashes into buckets.
    * Similarity computation over all buckets: Time it takes to compute the similarity values between trajectories in the same bucket for all buckets.

    Raises:
        ValueError: If measure not exists
    """

    # File handling
    scheme = "grid" if "grid" in measure else "disk"
  
    hash_generation_times_bucketing = []
    hash_generation_times_compression = []
    bucket_distribution_times = []
    all_similarity_runtimes = []

    with Pool(parallel_jobs) as pool:

        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")

            
            # --- Parallel Hashing ---
            hash_results = pool.starmap(
                compute_hash_parallel, 
                [
                    (measure, city, data_size, diameter_bucketing, layers_bucketing, disks_bucketing, res_bucketing)
                    for _ in range(parallel_jobs)
                ]
            )
            comp_hash_results = pool.starmap(
                compute_hash_parallel, 
                [
                    (measure, city, data_size, diameter_compression, layers_compression, disks_compression, res_compression)
                    for _ in range(parallel_jobs)
                ]
            )

            # Extract hashes and compute average hashing time
            hashes1_list, hashing_times1 = zip(*hash_results)  # Hashes for bucketing
            hashes2_list, hashing_times2 = zip(*comp_hash_results)  # Hashes for compression

            avg_hashing_bucketing_time = sum(hashing_times1) / parallel_jobs  # Compute average for bucketing
            avg_hashing_compression_time = sum(hashing_times2) / parallel_jobs  # Compute average for compression

            hash_generation_times_bucketing.append(avg_hashing_bucketing_time)
            hash_generation_times_compression.append(avg_hashing_compression_time)

            # --- Parallel Bucketing ---
            bucket_results = pool.starmap(
                timed_bucketing, [(h, bucketing_method) for h in hashes1_list]
            )

            # Extract bucket systems and compute average bucketing time
            bucket_systems, individual_bucketing_times = zip(*bucket_results)
            avg_bucket_time = sum(individual_bucketing_times) / parallel_jobs  # Compute average
            bucket_distribution_times.append(avg_bucket_time)

            # --- Parallel Similarity Computation ---
            if "dtw" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing, 
                    [(hashes2_list[i], scheme, "dtw", bucket_systems[i], False) for i in range(parallel_jobs)]
                )
            elif "frechet" in measure:
                execution_times = pool.starmap(
                    measure_hashed_cy_bucketing, 
                    [(hashes2_list[i], scheme, "frechet", bucket_systems[i], False) for i in range(parallel_jobs)]
                )

            # Collect all runtime values

            avg_similarity_time = sum([element[0] for element in execution_times]) / parallel_jobs # Compute average
            all_similarity_runtimes.append(avg_similarity_time)

            
    # Compute overall average runtime across all iterations and jobs
    avg_sim_comp_time = format(sum(all_similarity_runtimes) / len(all_similarity_runtimes), ".3f")
    avg_hash_bucketing_time = format(sum(hash_generation_times_bucketing) / len(hash_generation_times_bucketing), ".3f")
    avg_hash_compression_time = format(sum(hash_generation_times_compression) / len(hash_generation_times_compression), ".3f")
    avg_bucket_time = format(sum(bucket_distribution_times) / len(bucket_distribution_times), ".3f")

    #Total time
    total_time = format(float(avg_sim_comp_time) + float(avg_hash_bucketing_time) + float(avg_hash_compression_time) + float(avg_bucket_time), ".3f")

    # Create a final DataFrame with one row
    df_final = pd.DataFrame(
        {
            "Data Size": [data_size],
            "Average Similarity Computation Time (Seconds)": [avg_sim_comp_time],
            "Average Hash Generation Time (Seconds) - Bucketing": [avg_hash_bucketing_time],
            "Average Hash Generation Time (Seconds) - Compression": [avg_hash_compression_time],
            "Average Bucket Distribution Time (Seconds)": [avg_bucket_time],
            "Total time (Seconds)": [total_time],
        }
    )

    return df_final


###### New true runtime functions
def measure_true_similarities_v2(
    measure: str, data_folder: str, meta_file: str, parallel_jobs: int = 10
):
    """Common method for measuring the efficiency of the similarity algorithms"""
    files = mfh.read_meta_file(meta_file)
    trajectories = fh.load_trajectory_files(files, data_folder)

    with Pool(parallel_jobs) as pool:
        result = pool.map(
            sim[measure], [[trajectories, 1, 1] for _ in range(parallel_jobs)]
        )
    return result

def compute_true_similarity_runtimes_v2(
    city: str,
    measure: str,
    data_size: int, 
    parallel_jobs: int = 10,
    iterations: int = 3
):
    
    #Path to data folder
    data_folder = get_dataset_path(city)
    all_runtimes = []
    
    for iteration in range(iterations):  # Loop through each iteration
        print(f"Iteration {iteration+1}/{iterations}")
        meta_file = f"{data_folder}META-{data_size}.txt"
        
        #executions times
        execution_times = measure_true_similarities_v2(
            measure=measure,
            data_folder=data_folder,
            meta_file=meta_file,
            parallel_jobs=parallel_jobs,
        )
        
        #Extend outer list with times from each iteration
        all_runtimes.extend([element[0] for element in execution_times])
        
    # Compute overall average runtime across all iterations and jobs
    overall_avg_runtime = format(sum(all_runtimes) / len(all_runtimes), ".3f")
    
    # Create a final DataFrame with one row
    df_final = pd.DataFrame(
        {
            "Data Size": [data_size],
            "Average Similarity Computation Time (Seconds)": [overall_avg_runtime],
        }
    )
    
    return df_final