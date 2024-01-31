"""
Sheet containing method for creating, fetching and deleting metafiles
"""

import os, shutil
import re
import random
import numpy as np


def create_meta_files(
    path_to_files: str,
    data_prefix: str,
    prefix: str = "META",
    number: int = 100,
    create_test_set: bool = False,
    test_set_size: int = 50,
) -> None:
    """
    Function that creates metafiles (data-sets) with an incresing number of files containing random itemes from the chosen data.

    Parameters
    ----------
    path_to_files : str
        The path to the data folder
    data_prefix: str
        The prefix of the data that will be used in the sets
    prefix: str (default "META")
        The prefix of the meta_files
    number: int (default 100)
        The number of files that should be added to each set
    create_test_set: boolean (default True)
        Set this to true if a test set should be generated
    test_set_size: int (default 50)
        Preferred size of the testset
    """

    file_list = [
        file
        for file in os.listdir(path_to_files)
        if re.match(r"\b" + re.escape(data_prefix) + r"[^\\]*\.txt$", file)
    ]

    random.shuffle(file_list)

    data_set_sizes = np.arange(number, len(file_list) + 1, number).tolist()
    if create_test_set:
        data_set_sizes.append(test_set_size)

    for size in data_set_sizes:
        si = 0
        with open(f"{path_to_files}/{prefix}-{size}.txt", "w") as file:
            for file_name in file_list[0:size]:
                file.write("%s\n" % (file_name))
                si += 1
            file.close()

    return


def get_meta_files(path_to_files: str, prefix: str = "META") -> list:
    """
    Function that returns the metafiles in the given folder

    Parameters
    ----------
    path_to_files : str
        The path to the data folder
    prefix: str (default "META")
        The prefix of the meta_files that will be retrieved
    """

    file_list = [
        file
        for file in os.listdir(path_to_files)
        if re.match(r"\b" + re.escape(prefix) + r"[^\\]*\.txt$", file)
    ]

    return file_list


def delete_meta_files(path_to_files: str, prefix: str = "META") -> None:
    """
    Function that deletes the metafiles in the given folder

    Parameters
    ----------
    path_to_files : str
        The path to the data folder
    prefix: str (default "META")
        The prefix of the meta_files that will be retrieved
    """

    file_list = [
        file
        for file in os.listdir(path_to_files)
        if re.match(r"\b" + re.escape(prefix) + r"[^\\]*\.txt$", file)
    ]

    for filename in file_list:
        file_path = os.path.join(path_to_files, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to remove %s. Reason: %s" % (file_path, e))

    return


def read_meta_file(path_to_file: str) -> list[str]:
    """
    Reads and returns the content of a trajectory metafile as a list

    Parameters
    ---
    path_to_file : str
        Path to the metafile to be read

    Returns
    ---
    A list containing the filenames in the metafile
    """
    try:
        with open(path_to_file, "r") as file:
            trajectory_files = [line.rstrip() for line in file]
            file.close()
    except FileNotFoundError:
        raise (Exception(f"Cant find file {path_to_file}"))

    return trajectory_files
