from multiprocessing import Pool, pool
import pandas as pd

from computation.similarity import generate_disk_hash_similarity_with_bucketing, generate_disk_hash_similarity_with_bucketing_hybrid, generate_disk_hash_similarity_with_bucketing_with_true_sim, generate_grid_hash_similarity_with_bucketing, generate_grid_hash_similarity_with_bucketing_hybrid, generate_grid_hash_similarity_with_bucketing_with_true_sim
from result_analysis.disk_correlation_bucketing import fun_wrapper_corr_bucketing
from schemes.lsh_bucketing import *
from utils.helpers.bucket_evaluation import calculate_false_negatives, calculate_false_positives, calculate_true_positives, compute_bucket_system_f1_score, compute_bucket_system_precision, compute_bucket_system_recall, evaluate_bucket_system, evaluate_bucket_system_to_list, find_predicted_similar_trajectories, get_nearest_neighbour_under_threshold

def disk_compute_single_evaluation_scores(city, diameter, layers, disks, measure, size, true_trajectories, true_sim_matrix, bucketing_method, THRESHOLDS):
    
    if true_trajectories:
        hashed_similarities, bucket_system = generate_disk_hash_similarity_with_bucketing_with_true_sim(
            city=city, diameter=diameter, layers=layers, disks=disks, measure=measure, size=size, bucketing_method=bucketing_method
        )
        corr = 1
    else:
        hashed_similarities, bucket_system, total_comparisons, total_skipped_comparisons, num_compared_trajectories = generate_disk_hash_similarity_with_bucketing(
            city=city, diameter=diameter, layers=layers, disks=disks, measure=measure, size=size, bucketing_method=bucketing_method
        )
        corr = fun_wrapper_corr_bucketing(hashed_similarities, true_sim_matrix)
    
    
    # All bucket evaluations scores 
    bucket_evaluation = evaluate_bucket_system_to_list(bucket_system)
 
    # Precision, recall, f1 score for each threshold   
    threshold_results_for_p_r_f1 ={}

    # Loop through each threshold
    for threshold in THRESHOLDS:
        all_trajectory_names = list(hashed_similarities.keys())  # List of trajectories
        true_positives, false_positives, false_negatives = 0, 0, 0

        # Compute precision, recall, and F1 score
        for trajectory in all_trajectory_names:
            predicted_similar = find_predicted_similar_trajectories(trajectory, bucket_system)
            ground_truth = get_nearest_neighbour_under_threshold(trajectory, threshold, true_sim_matrix).index.to_list()
            true_positives += calculate_true_positives(predicted_similar, ground_truth)
            false_positives += calculate_false_positives(predicted_similar, ground_truth)
            false_negatives += calculate_false_negatives(predicted_similar, ground_truth)

        # Compute final scores
        precision = compute_bucket_system_precision(true_positives, false_positives)
        recall = compute_bucket_system_recall(true_positives, false_negatives)
        f1_score = compute_bucket_system_f1_score(precision, recall)
        
        threshold_results_for_p_r_f1[threshold] = [precision, recall, f1_score]
        
    return threshold_results_for_p_r_f1, bucket_evaluation, corr, total_comparisons, total_skipped_comparisons, num_compared_trajectories

def disk_compute_evaluation_scores(
    CITY, 
    MEASURE,
    DIAMETER,
    LAYERS,
    DISKS,
    SIZE,
    BUCKETING_METHOD,
    TRUE_TRAJECTORIES,
    TRUE_SIM_MATRIX,
    TRESHOLDS,
    parallell_jobs: int = 8,
    iterations : int = 3
):
        
    # Fina results
    all_total_buckets = 0
    all_largest_bucket_sizes = 0
    all_smallest_bucket_sizes = 0
    all_buckets_with_multiple_trajectories = 0
    all_buckets_with_single_trajectory = 0
    all_percentage_buckets_with_several_trajectories = 0
    all_percentage_buckets_with_single_trajectory = 0
    all_correlations = 0
    threshold_results ={threshold: [0,0,0] for threshold in TRESHOLDS}

    ##Legge til aggregering
    all_total_comparisons = 0
    all_total_skipped_comparisons = 0
    all_num_compared_trajectories = 0
    
    #Iterations
    with Pool(parallell_jobs) as pool:
        
        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")
                    
            evaluation_scores = pool.starmap(
                    disk_compute_single_evaluation_scores, [(CITY, DIAMETER, LAYERS, DISKS, MEASURE, SIZE, TRUE_TRAJECTORIES, TRUE_SIM_MATRIX, BUCKETING_METHOD, TRESHOLDS) for _ in range(parallell_jobs)]
            )
                
                
            # Extract hashes and compute average hashing time
            threshold_results_for_p_r_f1, bucket_evaluation, corr, total_comparisons, total_skipped_comparisons, num_compared_trajectories = zip(*evaluation_scores)  # Unpack results
                
                
            for threshold in TRESHOLDS:
                # Go through each parallel job's dictionary
                for i in range(parallell_jobs):
                    p, r, f1 = threshold_results_for_p_r_f1[i][threshold]
                    
                    # Add each value to the cumulative totals
                    threshold_results[threshold][0] += p
                    threshold_results[threshold][1] += r
                    threshold_results[threshold][2] += f1
                
            for i in range(parallell_jobs):
                all_total_buckets += bucket_evaluation[i][0]
                all_largest_bucket_sizes += bucket_evaluation[i][1]
                all_smallest_bucket_sizes += bucket_evaluation[i][2]
                all_buckets_with_multiple_trajectories += bucket_evaluation[i][3]
                all_buckets_with_single_trajectory += bucket_evaluation[i][4]
                all_percentage_buckets_with_several_trajectories += bucket_evaluation[i][5]
                all_percentage_buckets_with_single_trajectory += bucket_evaluation[i][6]
                all_correlations += corr[i]

                #New aggregering
                all_total_comparisons += total_comparisons[i]
                all_total_skipped_comparisons += total_skipped_comparisons[i]
                all_num_compared_trajectories += num_compared_trajectories[i]
        
    # for each threshold, divide by the number of parallel jobs * iterations
    for threshold in TRESHOLDS:
        threshold_results[threshold][0] /= parallell_jobs * iterations
        threshold_results[threshold][1] /= parallell_jobs * iterations
        threshold_results[threshold][2] /= parallell_jobs * iterations
            
    
    # Divide by the number of parallel jobs * iterations
    all_total_buckets /= parallell_jobs * iterations
    all_largest_bucket_sizes /= parallell_jobs * iterations
    all_smallest_bucket_sizes /= parallell_jobs * iterations
    all_buckets_with_multiple_trajectories /= parallell_jobs * iterations
    all_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_percentage_buckets_with_several_trajectories /= parallell_jobs * iterations
    all_percentage_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_correlations /= parallell_jobs * iterations

    #New aggregering
    all_total_comparisons /= parallell_jobs * iterations
    all_total_skipped_comparisons /= parallell_jobs * iterations
    all_num_compared_trajectories /= parallell_jobs * iterations
    
    rows = []
    for threshold in TRESHOLDS:
        precision_val = threshold_results[threshold][0]
        recall_val    = threshold_results[threshold][1]
        f1_val        = threshold_results[threshold][2]
        
        row_dict = {
            "City": CITY,
            "Measure": MEASURE,
            "Diameter": DIAMETER,
            "Layers": LAYERS,
            "Disks": DISKS,
            "Size": SIZE,
            "Threshold": threshold,
            "Avg Precision": format(precision_val, ".3f"),
            "Avg Recall": format(recall_val, ".3f"),
            "Avg F1 Score": format(f1_val, ".3f"),
            "Avg Total Buckets": format(all_total_buckets, ".3f"),
            "Avg Largest Bucket Size": format(all_largest_bucket_sizes, ".3f"),
            "Avg Smallest Bucket Size": format(all_smallest_bucket_sizes, ".3f"),
            "Avg Buckets with >1 Trajectory": format(all_buckets_with_multiple_trajectories, ".3f"),
            "Avg Buckets with 1 Trajectory": format(all_buckets_with_single_trajectory, ".3f"),
            "Avg Percentage >1 Trajectory": format(all_percentage_buckets_with_several_trajectories, ".3f"),
            "Avg Percentage 1 Trajectory": format(all_percentage_buckets_with_single_trajectory, ".3f"),
            "Avg Correlation Coefficient": format(all_correlations, ".3f"),
            "Avg Total Comparisons": format(all_total_comparisons, ".3f"),
            "Avg Total Skipped Comparisons": format(all_total_skipped_comparisons, ".3f"),
            "Avg Num Compared Trajectories": format(all_num_compared_trajectories, ".3f")
        }
        rows.append(row_dict)

    # Create DataFrame
    df_result = pd.DataFrame(rows)

    # Ensure columns appear in the exact requested order:
    column_order = [
            "City", 
            "Measure", 
            "Diameter", 
            "Layers", 
            "Disks", 
            "Size",
            "Threshold", 
            "Avg Precision", 
            "Avg Recall", 
            "Avg F1 Score",
            "Avg Total Buckets", 
            "Avg Largest Bucket Size", 
            "Avg Smallest Bucket Size",
            "Avg Buckets with >1 Trajectory", 
            "Avg Buckets with 1 Trajectory",
            "Avg Percentage >1 Trajectory", 
            "Avg Percentage 1 Trajectory",
            "Avg Correlation Coefficient",
            "Avg Total Comparisons",
            "Avg Total Skipped Comparisons",
            "Avg Num Compared Trajectories"
    ]
    df_result = df_result[column_order]

    return df_result

def grid_compute_single_evaluation_scores(city, resolution, layers, measure, size, true_trajectories, true_sim_matrix, bucketing_method, THRESHOLDS):
    
    if true_trajectories:
        hashed_similarities, bucket_system = generate_grid_hash_similarity_with_bucketing_with_true_sim(
            city=city, res=resolution, layers=layers, measure=measure, size=size, bucketing_method=bucketing_method
        )
        corr = 1
    else:
        hashed_similarities, bucket_system, total_comparisons, total_skipped_comparisons, num_compared_trajectories = generate_grid_hash_similarity_with_bucketing(
            city=city, res=resolution, layers=layers, measure=measure, size=size, bucketing_method=bucketing_method
        )
        corr = fun_wrapper_corr_bucketing(hashed_similarities, true_sim_matrix)
    
    
    # All bucket evaluations scores 
    bucket_evaluation = evaluate_bucket_system_to_list(bucket_system)
 
    # Precision, recall, f1 score for each threshold   
    threshold_results_for_p_r_f1 ={}

    # Loop through each threshold
    for threshold in THRESHOLDS:
        all_trajectory_names = list(hashed_similarities.keys())  # List of trajectories
        true_positives, false_positives, false_negatives = 0, 0, 0

        # Compute precision, recall, and F1 score
        for trajectory in all_trajectory_names:
            predicted_similar = find_predicted_similar_trajectories(trajectory, bucket_system)
            ground_truth = get_nearest_neighbour_under_threshold(trajectory, threshold, true_sim_matrix).index.to_list()
            true_positives += calculate_true_positives(predicted_similar, ground_truth)
            false_positives += calculate_false_positives(predicted_similar, ground_truth)
            false_negatives += calculate_false_negatives(predicted_similar, ground_truth)

        # Compute final scores
        precision = compute_bucket_system_precision(true_positives, false_positives)
        recall = compute_bucket_system_recall(true_positives, false_negatives)
        f1_score = compute_bucket_system_f1_score(precision, recall)
        
        threshold_results_for_p_r_f1[threshold] = [precision, recall, f1_score]
        
    return threshold_results_for_p_r_f1, bucket_evaluation, corr, total_comparisons, total_skipped_comparisons, num_compared_trajectories

def grid_compute_evaluation_scores(
    CITY, 
    MEASURE,
    RESOLUTION,
    LAYERS,
    SIZE,
    BUCKETING_METHOD,
    TRUE_TRAJECTORIES,
    TRUE_SIM_MATRIX,
    TRESHOLDS,
    parallell_jobs: int = 8,
    iterations : int = 3
):
        
    # Final results
    all_total_buckets = 0
    all_largest_bucket_sizes = 0
    all_smallest_bucket_sizes = 0
    all_buckets_with_multiple_trajectories = 0
    all_buckets_with_single_trajectory = 0
    all_percentage_buckets_with_several_trajectories = 0
    all_percentage_buckets_with_single_trajectory = 0
    all_correlations = 0
    threshold_results = {threshold: [0,0,0] for threshold in TRESHOLDS}
    
    ##Legge til aggregering
    all_total_comparisons = 0
    all_total_skipped_comparisons = 0
    all_num_compared_trajectories = 0
    
    #Iterations
    with Pool(parallell_jobs) as pool:
        for iteration in range(iterations):
            print(f"Iteration {iteration+1}/{iterations}")
                    
            evaluation_scores = pool.starmap(
                grid_compute_single_evaluation_scores, [(CITY, RESOLUTION, LAYERS, MEASURE, SIZE, TRUE_TRAJECTORIES, TRUE_SIM_MATRIX, BUCKETING_METHOD, TRESHOLDS) for _ in range(parallell_jobs)]
            )
            
            
            # Extract hashes and compute average hashing time
            threshold_results_for_p_r_f1, bucket_evaluation, corr, total_comparisons, total_skipped_comparisons, num_compared_trajectories = zip(*evaluation_scores)  # Unpack results
            
            
            for threshold in TRESHOLDS:
                # Go through each parallel job's dictionary
                for i in range(parallell_jobs):
                    p, r, f1 = threshold_results_for_p_r_f1[i][threshold]
                    
                    # Add each value to the cumulative totals
                    threshold_results[threshold][0] += p
                    threshold_results[threshold][1] += r
                    threshold_results[threshold][2] += f1
            
            for i in range(parallell_jobs):
                all_total_buckets += bucket_evaluation[i][0]
                all_largest_bucket_sizes += bucket_evaluation[i][1]
                all_smallest_bucket_sizes += bucket_evaluation[i][2]
                all_buckets_with_multiple_trajectories += bucket_evaluation[i][3]
                all_buckets_with_single_trajectory += bucket_evaluation[i][4]
                all_percentage_buckets_with_several_trajectories += bucket_evaluation[i][5]
                all_percentage_buckets_with_single_trajectory += bucket_evaluation[i][6]
                all_correlations += corr[i]
                
                #New aggregering
                all_total_comparisons += total_comparisons[i]
                all_total_skipped_comparisons += total_skipped_comparisons[i]
                all_num_compared_trajectories += num_compared_trajectories[i]
        
    # for each threshold, divide by the number of parallel jobs * iterations
    for threshold in TRESHOLDS:
        threshold_results[threshold][0] /= parallell_jobs * iterations
        threshold_results[threshold][1] /= parallell_jobs * iterations
        threshold_results[threshold][2] /= parallell_jobs * iterations
            
    
    # Divide by the number of parallel jobs * iterations
    all_total_buckets /= parallell_jobs * iterations
    all_largest_bucket_sizes /= parallell_jobs * iterations
    all_smallest_bucket_sizes /= parallell_jobs * iterations
    all_buckets_with_multiple_trajectories /= parallell_jobs * iterations
    all_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_percentage_buckets_with_several_trajectories /= parallell_jobs * iterations
    all_percentage_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_correlations /= parallell_jobs * iterations
    
    #New aggregering
    all_total_comparisons /= parallell_jobs * iterations
    all_total_skipped_comparisons /= parallell_jobs * iterations
    all_num_compared_trajectories /= parallell_jobs * iterations
    
    rows = []
    for threshold in TRESHOLDS:
        precision_val = threshold_results[threshold][0]
        recall_val    = threshold_results[threshold][1]
        f1_val        = threshold_results[threshold][2]
        
        row_dict = {
            "City": CITY,
            "Measure": MEASURE,
            "Resolution": RESOLUTION,
            "Layers": LAYERS,
            "Size": SIZE,
            "Threshold": threshold,
            "Avg Precision": format(precision_val, ".3f"),
            "Avg Recall": format(recall_val, ".3f"),
            "Avg F1 Score": format(f1_val, ".3f"),
            "Avg Total Buckets": format(all_total_buckets, ".3f"),
            "Avg Largest Bucket Size": format(all_largest_bucket_sizes, ".3f"),
            "Avg Smallest Bucket Size": format(all_smallest_bucket_sizes, ".3f"),
            "Avg Buckets with >1 Trajectory": format(all_buckets_with_multiple_trajectories, ".3f"),
            "Avg Buckets with 1 Trajectory": format(all_buckets_with_single_trajectory, ".3f"),
            "Avg Percentage >1 Trajectory": format(all_percentage_buckets_with_several_trajectories, ".3f"),
            "Avg Percentage 1 Trajectory": format(all_percentage_buckets_with_single_trajectory, ".3f"),
            "Avg Correlation Coefficient": format(all_correlations, ".3f"),
            "Avg Total Comparisons": format(all_total_comparisons, ".3f"),
            "Avg Total Skipped Comparisons": format(all_total_skipped_comparisons, ".3f"),
            "Avg Num Compared Trajectories": format(all_num_compared_trajectories, ".3f")
        }
        rows.append(row_dict)

    # Create DataFrame
    df_result = pd.DataFrame(rows)

    # Ensure columns appear in the exact requested order:
    column_order = [
            "City", 
            "Measure", 
            "Resolution", 
            "Layers", 
            "Size",
            "Threshold", 
            "Avg Precision", 
            "Avg Recall", 
            "Avg F1 Score",
            "Avg Total Buckets", 
            "Avg Largest Bucket Size", 
            "Avg Smallest Bucket Size",
            "Avg Buckets with >1 Trajectory", 
            "Avg Buckets with 1 Trajectory",
            "Avg Percentage >1 Trajectory", 
            "Avg Percentage 1 Trajectory",
            "Avg Correlation Coefficient",
            "Avg Total Comparisons",
            "Avg Total Skipped Comparisons",
            "Avg Num Compared Trajectories"
    ]
    df_result = df_result[column_order]

    return df_result



# Hybrid

def disk_hybrid_compute_single_evaluation_scores(
    city,
    diameter_bucketing,
    layers_bucketing,
    disks_bucketing,
    diameter_compression,
    layers_compression,
    disks_compression,
    measure,
    size,
    true_sim_matrix,
    bucketing_method,
    THRESHOLDS):
    
    # Generate bucket system and hashed similarities
    hashed_similarities, bucket_system = generate_disk_hash_similarity_with_bucketing_hybrid(
        city=city, 
        diameter_bucketing=diameter_bucketing,
        diameter_compression=diameter_compression, 
        layers_bucketing=layers_bucketing,
        layers_compression=layers_compression,
        disks_bucketing=disks_bucketing,
        disks_compression=disks_compression, 
        measure=measure,
        size=size,
        bucketing_method=bucketing_method
    )
    
    #Correlation
    corr = fun_wrapper_corr_bucketing(hashed_similarities, true_sim_matrix)
    
    
    # All bucket evaluations scores 
    bucket_evaluation = evaluate_bucket_system_to_list(bucket_system)
 
    # Precision, recall, f1 score for each threshold   
    threshold_results_for_p_r_f1 ={}

    # Loop through each threshold
    for threshold in THRESHOLDS:
        all_trajectory_names = list(hashed_similarities.keys())  # List of trajectories
        true_positives, false_positives, false_negatives = 0, 0, 0

        # Compute precision, recall, and F1 score
        for trajectory in all_trajectory_names:
            predicted_similar = find_predicted_similar_trajectories(trajectory, bucket_system)
            ground_truth = get_nearest_neighbour_under_threshold(trajectory, threshold, true_sim_matrix).index.to_list()
            true_positives += calculate_true_positives(predicted_similar, ground_truth)
            false_positives += calculate_false_positives(predicted_similar, ground_truth)
            false_negatives += calculate_false_negatives(predicted_similar, ground_truth)

        # Compute final scores
        precision = compute_bucket_system_precision(true_positives, false_positives)
        recall = compute_bucket_system_recall(true_positives, false_negatives)
        f1_score = compute_bucket_system_f1_score(precision, recall)
        
        threshold_results_for_p_r_f1[threshold] = [precision, recall, f1_score]
        
    return threshold_results_for_p_r_f1, bucket_evaluation, corr

def disk_hybrid_compute_evaluation_scores(
    CITY, 
    DIAMETER_BUCKETING,
    LAYERS_BUCKETING,
    DISKS_BUCKETING,
    DIAMETER_COMPRESSION,
    LAYERS_COMPRESSION,
    DISKS_COMPRESSION,
    MEASURE,
    SIZE,
    BUCKETING_METHOD,
    TRUE_SIM_MATRIX,
    TRESHOLDS,
    parallell_jobs: int = 8,
    iterations : int = 3
):
        
    # Final results
    all_total_buckets = 0
    all_largest_bucket_sizes = 0
    all_smallest_bucket_sizes = 0
    all_buckets_with_multiple_trajectories = 0
    all_buckets_with_single_trajectory = 0
    all_percentage_buckets_with_several_trajectories = 0
    all_percentage_buckets_with_single_trajectory = 0
    all_correlations = 0
    threshold_results ={threshold: [0,0,0] for threshold in TRESHOLDS}
    
    #Iterations
    for iteration in range(iterations):
        print(f"Iteration {iteration+1}/{iterations}")
                
        with Pool(parallell_jobs) as pool:
            evaluation_scores = pool.starmap(
                disk_hybrid_compute_single_evaluation_scores, [(CITY, DIAMETER_BUCKETING, LAYERS_BUCKETING, DISKS_BUCKETING, DIAMETER_COMPRESSION, LAYERS_COMPRESSION, DISKS_COMPRESSION, MEASURE, SIZE, TRUE_SIM_MATRIX, BUCKETING_METHOD, TRESHOLDS) for _ in range(parallell_jobs)]
            )
            
            
            # Extract hashes and compute average hashing time
            threshold_results_for_p_r_f1, bucket_evaluation, corr = zip(*evaluation_scores)  # Unpack results
            
            
            for threshold in TRESHOLDS:
                # Go through each parallel job's dictionary
                for i in range(parallell_jobs):
                    p, r, f1 = threshold_results_for_p_r_f1[i][threshold]
                    
                    # Add each value to the cumulative totals
                    threshold_results[threshold][0] += p
                    threshold_results[threshold][1] += r
                    threshold_results[threshold][2] += f1
            
            for i in range(parallell_jobs):
                all_total_buckets += bucket_evaluation[i][0]
                all_largest_bucket_sizes += bucket_evaluation[i][1]
                all_smallest_bucket_sizes += bucket_evaluation[i][2]
                all_buckets_with_multiple_trajectories += bucket_evaluation[i][3]
                all_buckets_with_single_trajectory += bucket_evaluation[i][4]
                all_percentage_buckets_with_several_trajectories += bucket_evaluation[i][5]
                all_percentage_buckets_with_single_trajectory += bucket_evaluation[i][6]
                all_correlations += corr[i]
        
    # for each threshold, divide by the number of parallel jobs * iterations
    for threshold in TRESHOLDS:
        threshold_results[threshold][0] /= parallell_jobs * iterations
        threshold_results[threshold][1] /= parallell_jobs * iterations
        threshold_results[threshold][2] /= parallell_jobs * iterations
            
    
    # Divide by the number of parallel jobs * iterations
    all_total_buckets /= parallell_jobs * iterations
    all_largest_bucket_sizes /= parallell_jobs * iterations
    all_smallest_bucket_sizes /= parallell_jobs * iterations
    all_buckets_with_multiple_trajectories /= parallell_jobs * iterations
    all_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_percentage_buckets_with_several_trajectories /= parallell_jobs * iterations
    all_percentage_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_correlations /= parallell_jobs * iterations
    
    rows = []
    for threshold in TRESHOLDS:
        precision_val = threshold_results[threshold][0]
        recall_val    = threshold_results[threshold][1]
        f1_val        = threshold_results[threshold][2]
        
        row_dict = {
            "City": CITY,
            "Measure": MEASURE,
            "Dia_buck": DIAMETER_BUCKETING,
            "Lay_buck": LAYERS_BUCKETING,
            "Dis_buck": DISKS_BUCKETING,
            "Dia_comp": DIAMETER_COMPRESSION,
            "Lay_comp": LAYERS_COMPRESSION,
            "Dis_comp": DISKS_COMPRESSION,
            "Size": SIZE,
            "Threshold": threshold,
            "Avg Precision": format(precision_val, ".3f"),
            "Avg Recall": format(recall_val, ".3f"),
            "Avg F1 Score": format(f1_val, ".3f"),
            "Avg Total Buckets": format(all_total_buckets, ".3f"),
            "Avg Largest Bucket Size": format(all_largest_bucket_sizes, ".3f"),
            "Avg Smallest Bucket Size": format(all_smallest_bucket_sizes, ".3f"),
            "Avg Buckets with >1 Trajectory": format(all_buckets_with_multiple_trajectories, ".3f"),
            "Avg Buckets with 1 Trajectory": format(all_buckets_with_single_trajectory, ".3f"),
            "Avg % >1 Trajectory": format(all_percentage_buckets_with_several_trajectories, ".3f"),
            "Avg % 1 Trajectory": format(all_percentage_buckets_with_single_trajectory, ".3f"),
            "Avg Correlation Coefficient": format(all_correlations, ".3f")
        }
        rows.append(row_dict)

    # Create DataFrame
    df_result = pd.DataFrame(rows)

    # Ensure columns appear in the exact requested order:
    column_order = [
            "City", 
            "Measure", 
            "Dia_buck",
            "Lay_buck",
            "Dis_buck",
            "Dia_comp",
            "Lay_comp",
            "Dis_comp", 
            "Size",
            "Threshold", 
            "Avg Precision", 
            "Avg Recall", 
            "Avg F1 Score",
            "Avg Total Buckets", 
            "Avg Largest Bucket Size", 
            "Avg Smallest Bucket Size",
            "Avg Buckets with >1 Trajectory", 
            "Avg Buckets with 1 Trajectory",
            "Avg % >1 Trajectory", 
            "Avg % 1 Trajectory",
            "Avg Correlation Coefficient"
    ]
    df_result = df_result[column_order]

    return df_result

def grid_hybrid_compute_single_evaluation_scores(
    city,
    resolution_bucketing,
    layers_bucketing,
    resolution_compression,
    layers_compression,
    measure,
    size,
    true_sim_matrix,
    bucketing_method,
    THRESHOLDS):
    
    # Generate bucket system and hashed similarities
    hashed_similarities, bucket_system = generate_grid_hash_similarity_with_bucketing_hybrid(
        city=city, 
        resolution_bucketing=resolution_bucketing,
        layers_bucketing=layers_bucketing,
        resolution_compression=resolution_compression,
        layers_compression=layers_compression,
        measure=measure,
        size=size,
        bucketing_method=bucketing_method
    )

    #Correlation
    corr = fun_wrapper_corr_bucketing(hashed_similarities, true_sim_matrix)
    
    
    # All bucket evaluations scores 
    bucket_evaluation = evaluate_bucket_system_to_list(bucket_system)
 
    # Precision, recall, f1 score for each threshold   
    threshold_results_for_p_r_f1 ={}

    # Loop through each threshold
    for threshold in THRESHOLDS:
        all_trajectory_names = list(hashed_similarities.keys())  # List of trajectories
        true_positives, false_positives, false_negatives = 0, 0, 0

        # Compute precision, recall, and F1 score
        for trajectory in all_trajectory_names:
            predicted_similar = find_predicted_similar_trajectories(trajectory, bucket_system)
            ground_truth = get_nearest_neighbour_under_threshold(trajectory, threshold, true_sim_matrix).index.to_list()
            true_positives += calculate_true_positives(predicted_similar, ground_truth)
            false_positives += calculate_false_positives(predicted_similar, ground_truth)
            false_negatives += calculate_false_negatives(predicted_similar, ground_truth)

        # Compute final scores
        precision = compute_bucket_system_precision(true_positives, false_positives)
        recall = compute_bucket_system_recall(true_positives, false_negatives)
        f1_score = compute_bucket_system_f1_score(precision, recall)
        
        threshold_results_for_p_r_f1[threshold] = [precision, recall, f1_score]
        
    return threshold_results_for_p_r_f1, bucket_evaluation, corr

def grid_hybrid_compute_evaluation_scores(
    CITY, 
    RESOLUTION_BUCKETING,
    LAYERS_BUCKETING,
    RESOLUTION_COMPRESSION,
    LAYERS_COMPRESSION,
    MEASURE,
    SIZE,
    BUCKETING_METHOD,
    TRUE_SIM_MATRIX,
    TRESHOLDS,
    parallell_jobs: int = 8,
    iterations : int = 3
):
        
    # Final results
    all_total_buckets = 0
    all_largest_bucket_sizes = 0
    all_smallest_bucket_sizes = 0
    all_buckets_with_multiple_trajectories = 0
    all_buckets_with_single_trajectory = 0
    all_percentage_buckets_with_several_trajectories = 0
    all_percentage_buckets_with_single_trajectory = 0
    all_correlations = 0
    threshold_results ={threshold: [0,0,0] for threshold in TRESHOLDS}
    
    #Iterations
    for iteration in range(iterations):
        print(f"Iteration {iteration+1}/{iterations}")
                
        with Pool(parallell_jobs) as pool:
            evaluation_scores = pool.starmap(
                grid_hybrid_compute_single_evaluation_scores, [(CITY, RESOLUTION_BUCKETING, LAYERS_BUCKETING, RESOLUTION_COMPRESSION, LAYERS_COMPRESSION, MEASURE, SIZE, TRUE_SIM_MATRIX, BUCKETING_METHOD, TRESHOLDS) for _ in range(parallell_jobs)]
            )
            
            
            # Extract hashes and compute average hashing time
            threshold_results_for_p_r_f1, bucket_evaluation, corr = zip(*evaluation_scores)  # Unpack results
            
            
            for threshold in TRESHOLDS:
                # Go through each parallel job's dictionary
                for i in range(parallell_jobs):
                    p, r, f1 = threshold_results_for_p_r_f1[i][threshold]
                    
                    # Add each value to the cumulative totals
                    threshold_results[threshold][0] += p
                    threshold_results[threshold][1] += r
                    threshold_results[threshold][2] += f1
            
            for i in range(parallell_jobs):
                all_total_buckets += bucket_evaluation[i][0]
                all_largest_bucket_sizes += bucket_evaluation[i][1]
                all_smallest_bucket_sizes += bucket_evaluation[i][2]
                all_buckets_with_multiple_trajectories += bucket_evaluation[i][3]
                all_buckets_with_single_trajectory += bucket_evaluation[i][4]
                all_percentage_buckets_with_several_trajectories += bucket_evaluation[i][5]
                all_percentage_buckets_with_single_trajectory += bucket_evaluation[i][6]
                all_correlations += corr[i]
        
    # for each threshold, divide by the number of parallel jobs * iterations
    for threshold in TRESHOLDS:
        threshold_results[threshold][0] /= parallell_jobs * iterations
        threshold_results[threshold][1] /= parallell_jobs * iterations
        threshold_results[threshold][2] /= parallell_jobs * iterations
            
    
    # Divide by the number of parallel jobs * iterations
    all_total_buckets /= parallell_jobs * iterations
    all_largest_bucket_sizes /= parallell_jobs * iterations
    all_smallest_bucket_sizes /= parallell_jobs * iterations
    all_buckets_with_multiple_trajectories /= parallell_jobs * iterations
    all_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_percentage_buckets_with_several_trajectories /= parallell_jobs * iterations
    all_percentage_buckets_with_single_trajectory /= parallell_jobs * iterations
    all_correlations /= parallell_jobs * iterations
    
    rows = []
    for threshold in TRESHOLDS:
        precision_val = threshold_results[threshold][0]
        recall_val    = threshold_results[threshold][1]
        f1_val        = threshold_results[threshold][2]
        
        row_dict = {
            "City": CITY,
            "Measure": MEASURE,
            "Res_buck": RESOLUTION_BUCKETING,
            "Lay_buck": LAYERS_BUCKETING,
            "Res_comp": RESOLUTION_COMPRESSION,
            "Lay_comp": LAYERS_COMPRESSION,
            "Size": SIZE,
            "Threshold": threshold,
            "Avg Precision": format(precision_val, ".3f"),
            "Avg Recall": format(recall_val, ".3f"),
            "Avg F1 Score": format(f1_val, ".3f"),
            "Avg Total Buckets": format(all_total_buckets, ".3f"),
            "Avg Largest Bucket Size": format(all_largest_bucket_sizes, ".3f"),
            "Avg Smallest Bucket Size": format(all_smallest_bucket_sizes, ".3f"),
            "Avg Buckets with >1 Trajectory": format(all_buckets_with_multiple_trajectories, ".3f"),
            "Avg Buckets with 1 Trajectory": format(all_buckets_with_single_trajectory, ".3f"),
            "Avg % >1 Trajectory": format(all_percentage_buckets_with_several_trajectories, ".3f"),
            "Avg % 1 Trajectory": format(all_percentage_buckets_with_single_trajectory, ".3f"),
            "Avg Correlation Coefficient": format(all_correlations, ".3f")
        }
        rows.append(row_dict)

    # Create DataFrame
    df_result = pd.DataFrame(rows)

    # Ensure columns appear in the exact requested order:
    column_order = [
            "City", 
            "Measure", 
            "Res_buck",
            "Lay_buck",
            "Res_comp",
            "Lay_comp",
            "Size",
            "Threshold", 
            "Avg Precision", 
            "Avg Recall", 
            "Avg F1 Score",
            "Avg Total Buckets", 
            "Avg Largest Bucket Size", 
            "Avg Smallest Bucket Size",
            "Avg Buckets with >1 Trajectory", 
            "Avg Buckets with 1 Trajectory",
            "Avg % >1 Trajectory", 
            "Avg % 1 Trajectory",
            "Avg Correlation Coefficient"
    ]
    df_result = df_result[column_order]

    return df_result



### Helper function
def disk_gather_bucketing_and_compression_lists(params_dict):
    """
    Returns six lists in this order:
    (compression_diameters, compression_layers, compression_disks,
     bucketing_diameters,  bucketing_layers,   bucketing_disks)
    """
    comp_diameters = []
    comp_layers = []
    comp_disks = []
    buck_diameters = []
    buck_layers = []
    buck_disks = []

    for config_name, config_data in params_dict.items():
        # compression
        comp = config_data["compression"]
        comp_diameters.append(comp["diameter"])
        comp_layers.append(comp["layers"])
        comp_disks.append(comp["disks"])

        # bucketing
        buck = config_data["bucketing"]
        buck_diameters.append(buck["diameter"])
        buck_layers.append(buck["layers"])
        buck_disks.append(buck["disks"])

    return (
        comp_diameters,
        comp_layers,
        comp_disks,
        buck_diameters,
        buck_layers,
        buck_disks
)
    
def grid_gather_bucketing_and_compression_lists(params_dict):
    """
    Returns six lists in this order:
    (compression_resolutions, compression_layers,
        bucketing_resolutions,  bucketing_layers)
    """
    comp_resolutions = []
    comp_layers = []
    buck_resolutions = []
    buck_layers = []

    for config_name, config_data in params_dict.items():
        # compression
        comp = config_data["compression"]
        comp_resolutions.append(comp["resolution"])
        comp_layers.append(comp["layers"])

        # bucketing
        buck = config_data["bucketing"]
        buck_resolutions.append(buck["resolution"])
        buck_layers.append(buck["layers"])

    return (
        comp_resolutions,
        comp_layers,
        buck_resolutions,
        buck_layers,
    )