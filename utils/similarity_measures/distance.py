import pandas as pd
from utils.similarity_measures.hashed_frechet import (
    cy_frechet_hashes,
    cy_frechet_hashes_pool,
)
from utils.similarity_measures.hashed_dtw import cy_dtw_hashes, cy_dtw_hashes_pool


def transform_np_numerical_disk_hashes_to_non_np(
    hashes: dict[str, list[list[float]]]
) -> dict[str, list[list[list[float]]]]:
    """Transforms the numerical disk hashes to a format that fits the true dtw similarity measure (non numpy input)"""
    transformed_data = {
        key: [[array.tolist() for array in sublist] for sublist in value]
        for key, value in hashes.items()
    }
    return transformed_data


def compute_hash_similarity(
    hashes: dict[str, list[list[list[float]]]],
    scheme: str,
    measure: str,
    parallel: bool = False,
) -> pd.DataFrame:
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



def disk_coordinates(hashes: dict[str, list[list[float]]]) -> pd.DataFrame:
    """The hashed disk coordinates"""
    hashed_coordinates = transform_np_numerical_disk_hashes_to_non_np(hashes)
    return hashed_coordinates


#####################################################################################NEW CODE - BUCKETING########################


def compute_hash_similarity_within_buckets(
    hashes: dict[str, list[list[list[float]]]],
    scheme: str,
    measure: str,
    bucket_system: dict[int, list[str]],
    parallel: bool = False,
) -> pd.DataFrame:
    
    
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