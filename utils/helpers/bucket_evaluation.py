# Sheet containing helper functions for bucket evaluation
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


def calculate_true_negatives(predicted: list[str], ground_truth: list[str]):
    # Calculate the number of true negatives
    pass
    
    

def calculate_false_positives(predicted: list[str], ground_truth: list[str]):
    """
    Calculate the number of false positives between the predicted and ground truth values.
    """
    # Calculate the number of false positives
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
    