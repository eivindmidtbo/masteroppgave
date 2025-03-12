# Sheet containing helper functions for bucket evaluation
from decimal import DivisionImpossible
import pandas as pd


def get_nearest_neighbours(trajectory_name, number_of_neighbours, true_sim_matrix):
    """
    Retrieves the k number of nearest neighbours for a given trajectory.
    """
    # Get the similarity values for the given trajectory
    similarity_values = true_sim_matrix[trajectory_name]
    # Sort the values in ascending order
    sorted_values = similarity_values.sort_values()
    # Get the n most similar trajectories
    nearest_neighbours = sorted_values[1:number_of_neighbours+1]
    return nearest_neighbours


def get_nearest_neighbour_under_threshold(trajectory_name, threshold, true_sim_matrix):
    """
    Retrieves all the trajectories with a similarity value less than the threshold.
    """
    
    if trajectory_name not in true_sim_matrix.columns:
        raise ValueError(f"Trajectory '{trajectory_name}' not found in the DataFrame.")
    
    # Select the column based on the trajectory name
    filtered_column = true_sim_matrix[trajectory_name]
    
    # Filter values based on the threshold
    df_threshold = filtered_column[filtered_column < threshold]
    
    # Convert to Series with the same name as the trajectory
    series = pd.Series(df_threshold, name=trajectory_name)  
    
    # Remove self-reference (trajectory_name itself)
    filtered_series = series[series.index != trajectory_name]
    
    #sort the values in ascending order
    filtered_series = filtered_series.sort_values()
    
    return filtered_series


def calculate_true_positives(predicted: list[str], ground_truth: list[str]) -> int:
    """
    Calculate the number of true positives between the predicted and ground truth values.
    """
    return len(set(predicted).intersection(ground_truth))

def calculate_false_positives(predicted: list[str], ground_truth: list[str]) -> int:
    """
    Calculate the number of false positives between the predicted and ground truth values.
    If ground_truth is empty, all predicted values are considered false positives.
    """
    if not ground_truth:  # Check if ground_truth is empty
        return len(predicted)  # All predicted elements are false positives

    return len(set(predicted).difference(ground_truth))


def calculate_false_negatives(predicted: list[str], ground_truth: list[str]):
    """
    Calculate the number of false negatives between the predicted and ground truth values.
    """
    # Calculate the number of false negatives
    return len(set(ground_truth).difference(predicted))
    


def find_predicted_similar_trajectories(trajectory_name: str, bucket_system):
    """
    Find all the predicted similar trajectories for a given trajectory in the bucket system.
    """

    shared_trajectories = set()

    for trajectories in bucket_system.values():
        if trajectory_name in trajectories:
            shared_trajectories.update(trajectories)

    # Remove the original trajectory from the result
    shared_trajectories.discard(trajectory_name)
    
    return list(shared_trajectories)
    
    
    
def compute_bucket_system_precision(true_positives, false_positives):
    """
    Compute the precision of the bucket system.
    """
    if true_positives == 0 and false_positives == 0:
        return 0
    precision = true_positives / (true_positives + false_positives)
    return precision

def compute_bucket_system_recall(true_positives, false_negatives):
    """
    Compute the recall of the bucket system.
    """
    if true_positives == 0 and false_negatives == 0:
        return 0
    recall = true_positives / (true_positives + false_negatives)
    return recall

def compute_bucket_system_f1_score(precision, recall):
    """
    Compute the F1 score of the bucket system.
    """
    if precision == 0 and recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_bucket_system(bucket_system):
    """
    Analyzes the given bucket system and returns a DataFrame with key statistics.

    Parameters:
    - bucket_system (dict): A dictionary where keys are bucket IDs and values are lists of trajectories.

    Returns:
    - pandas.DataFrame: A DataFrame containing bucket statistics.
    """
    total_buckets = len(bucket_system)
    buckets_with_multiple = sum(1 for trajectories in bucket_system.values() if len(trajectories) > 1)
    buckets_with_single = total_buckets - buckets_with_multiple
    largest_bucket_size = max(len(trajectories) for trajectories in bucket_system.values())
    smallest_bucket_size = min(len(trajectories) for trajectories in bucket_system.values())

    # Compute distribution percentages
    multiple_bucket_percentage = (buckets_with_multiple / total_buckets) * 100 if total_buckets > 0 else 0
    single_bucket_percentage = (buckets_with_single / total_buckets) * 100 if total_buckets > 0 else 0

    # Creating DataFrame
    stats = {
        "Total Buckets": [total_buckets],
        "Largest Bucket Size": [largest_bucket_size],
        "Smallest Bucket Size": [smallest_bucket_size],
        "Buckets with >1 Trajectory": [buckets_with_multiple],
        "Buckets with 1 Trajectory": [buckets_with_single],
        "Percentage >1 Trajectory": [f"{multiple_bucket_percentage:.2f}%"],
        "Percentage 1 Trajectory": [f"{single_bucket_percentage:.2f}%"]
    }

    df = pd.DataFrame(stats)
    return df


def evaluate_bucket_system_to_list(bucket_system):
    """
    Analyzes the given bucket system and returns a DataFrame with key statistics.

    Parameters:
    - bucket_system (dict): A dictionary where keys are bucket IDs and values are lists of trajectories.

    Returns:
    - pandas.DataFrame: A DataFrame containing bucket statistics.
    """
    total_buckets = len(bucket_system)
    buckets_with_multiple = sum(1 for trajectories in bucket_system.values() if len(trajectories) > 1)
    buckets_with_single = total_buckets - buckets_with_multiple
    largest_bucket_size = max(len(trajectories) for trajectories in bucket_system.values())
    smallest_bucket_size = min(len(trajectories) for trajectories in bucket_system.values())

    # Compute distribution percentages
    multiple_bucket_percentage = (buckets_with_multiple / total_buckets) * 100 if total_buckets > 0 else 0
    single_bucket_percentage = (buckets_with_single / total_buckets) * 100 if total_buckets > 0 else 0

    results = [total_buckets, largest_bucket_size, smallest_bucket_size, buckets_with_multiple, buckets_with_single, multiple_bucket_percentage, single_bucket_percentage]
    return results