from multiprocessing import Pool
import os, sys
import pandas as pd

currentdir = os.path.dirname(os.path.abspath("__file__"))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.similarity_measures import dtw, frechet, hashed_dtw, hashed_frechet
from utils.similarity_measures.distance import compute_hash_similarity
from utils.helpers import file_handler as fh
from utils.helpers import metafile_handler as mfh
from hashed_similarities import grid_similarity, disk_similarity


def get_dataset_path(city: str) -> str:
    return f"../dataset/{city}/output/"


sim = {
    "true_dtw_cy": dtw.measure_cy_dtw,
    "true_frechet_cy": frechet.measure_cy_frechet,
    "disk_dtw_cy": hashed_dtw.measure_cy_dtw_hashes,
    "disk_frechet_cy": hashed_frechet.measure_cy_frechet_hashes,
    "grid_dtw_cy": hashed_dtw.measure_cy_dtw,
    "grid_frechet_cy": hashed_frechet.measure_cy_frechet,
}


def measure_similarities(
    measure: str, data_folder: str, meta_file: str, parallel_jobs: int = 10
):
    """Common method for measuring the efficiency of the similarity algorithms"""
    # if (
    #     measure == "disk_dtw_cy"
    #     or measure == "disk_frechet_cy"
    #     or measure == "grid_dtw_cy"
    #     or measure == "grid_frechet_cy"
    # ):
    #     if measure == "disk_dtw_cy" or measure == "disk_frechet_cy":
    #         scheme = "disk"
    #     elif measure == "grid_dtw_cy" or measure == "grid_frechet_cy":
    #         grid = grid_similarity._constructGrid(city, res, layers, size)
    #         hashes = grid.compute_dataset_hashes()
    #         similarities = compute_hash_similarity(
    #             hashes=hashes, scheme="grid", measure=measure, parallel=True
    #         )
    files = mfh.read_meta_file(meta_file)
    trajectories = fh.load_trajectory_files(files, data_folder)

    with Pool() as pool:
        result = pool.map(
            sim[measure], [[trajectories, 1, 1] for _ in range(parallel_jobs)]
        )
    return result


def write_similarity_runtimes(
    measure: str,
    city: str,
    parallel_jobs: int = 10,
    data_start_size: int = 100,
    data_end_size: int = 1000,
    data_step_size: int = 100,
):

    data_folder = get_dataset_path(city)

    """Writes the runtimes of the similarity measures to a csv file"""
    data_sets = range(data_start_size, data_end_size, data_step_size)

    output_folder = "../benchmarks/similarities_runtimes/"
    file_name = f"similarity_runtimes_{measure}_porto_start({data_start_size})_end({data_end_size})_step({data_step_size}).csv"

    df = pd.DataFrame(
        index=[f"run_{x+1}" for x in range(parallel_jobs)],
        columns=[x for x in data_sets],
    )
    print(f"Computing {measure} for {city} with {parallel_jobs} jobs")

    index = 1
    for size in data_sets:
        print(f"Computing size {size}, set {index}/{len(data_sets)}", end="\r")

        meta_file = data_folder + f"META-{size}.txt"
        execution_times = measure_similarities(
            measure=measure,
            data_folder=data_folder,
            meta_file=meta_file,
            parallel_jobs=parallel_jobs,
        )
        df[size] = [element[0] for element in execution_times]
        index += 1
    df.to_csv(os.path.join(output_folder, file_name))
    print(df)
