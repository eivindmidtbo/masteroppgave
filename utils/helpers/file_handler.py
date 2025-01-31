"""
Sheet containing methods for reading trajectories
"""

import os, re
import shutil
import ast


def read_trajectory_file(file_path: str) -> list[list[float]]:
    """
    Reads a trajectory.txt file and returns the content as a list of coordinates

    Parameters
    ----------
    file_path : str
        The file path for the file that should be read

    Returns
    ---
    A list containing the files' coordinates as floats
    """

    try:
        with open(file_path, "r") as file:
            trajectory = [list(map(float, line.rstrip().split(","))) for line in file]
            file.close()
    except FileNotFoundError:
        print("Can't find file.")

    return trajectory


def load_trajectory_files(files: list[str], folder_path) -> dict:
    """
    Loads all trajectory.txt files and returns the content as a dictionary

    Parameters
    ----------
    files : list[str]
        A list of the files that should be read

    Returns
    ---
    A dictionary containing the files and their coordinates with their filename as key
    """

    file_list = files
    trajectories = dict()

    for file_name in file_list:
        key = os.path.splitext(file_name)[0]
        trajectory = read_trajectory_file(folder_path + file_name)

        trajectories[key] = trajectory

    return trajectories


def load_all_trajectory_files(folder_path: str, prefix: str) -> dict:
    """
    Reads all trajectory.txt files with the given prefix in the folder and returns a dictionary containing the data

    Parameters
    ----------
    folder_path : str
        The file path for the file that should be read
    prefix : str
        The prefix of the files that should be loaded

    Returns
    ---
    A dictionary containing all files with their filename as key
    """

    file_list = [
        file
        for file in os.listdir(folder_path)
        if re.match(r"\b" + re.escape(prefix) + r"[^\\]*\.txt$", file)
    ]

    trajectories = dict()

    for file_name in file_list:
        key = os.path.splitext(file_name)[0]
        trajectory = read_trajectory_file(folder_path + file_name)

        trajectories[key] = trajectory

    return trajectories


def read_hash_file(file_path: str) -> list[list[float]]:
    """
    Reads a hash.txt file and returns the content as a list of hashes

    Parameters
    ----------
    file_path : str
        The file path for the file that should be read

    Returns
    ---
    A list containing the files' hashes as lists
    """

    try:
        with open(file_path, "r") as file:
            hashes = [
                line.replace(" ", "").replace("'", "")[1:-2].split(",") for line in file
            ]
            file.close()
    except FileNotFoundError:
        print("Can't find file.")

    return hashes


def read_hash_file_to_float(file_path: str) -> list[list[float]]:
    """
    Reads a hash.txt file and returns the content as a list of hashes

    Parameters
    ----------
    file_path : str
        The file path for the file that should be read

    Returns
    ---
    A list containing the files' hashes as lists
    """

    hashes = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()  # Remove newline characters and spaces
                if line:  # Ensure the line is not empty
                    try:
                        parsed_list = ast.literal_eval(line)  # Safely convert string list to Python list
                        hashes.append(parsed_list)
                    except (SyntaxError, ValueError) as e:
                        print(f"Skipping invalid line: {line} - Error: {e}")
    except FileNotFoundError:
        print("Can't find file.")
    return hashes


def load_trajectory_hashes(files: list[str], folder_path: str) -> dict:
    """
    Loads all hashes.txt files and returns the content as a dictionary

    Parameters
    ----------
    files : list[str]
        A list of the files that should be read

    Returns
    ---
    A dictionary containing the files and their hashes with their filename as key
    """

    file_list = files
    hashes = dict()

    for file_name in file_list:
        key = os.path.splitext(file_name)[0]
        hash = read_hash_file(folder_path + file_name)

        hashes[key] = hash

    return hashes


def load_all_trajectory_hashes(folder_path: str, prefix: str) -> dict:
    """
    Reads all hash.txt files with the given prefix in the folder and returns a dictionary containing the data

    Parameters
    ----------
    folder_path : str
        The file path for the file that should be read
    prefix : str
        The prefix of the files that should be loaded

    Returns
    ---
    A dictionary containing all files with their filename as key
    """

    file_list = [
        file
        for file in os.listdir(folder_path)
        if re.match(r"\b" + re.escape(prefix) + r"[^\\]*\.txt$", file)
    ]

    hashes = dict()

    for file_name in file_list:
        key = os.path.splitext(file_name)[0]
        hashed_trajectory = read_hash_file_to_float(folder_path + file_name)

        hashes[key] = hashed_trajectory

    return hashes


def delete_old_files(output_folder: str, file_prefix_to_keep: str = None) -> None:
    for filename in os.listdir(output_folder):
        if file_prefix_to_keep is not None and filename.startswith(file_prefix_to_keep):
            continue
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to remove %s. Reason: %s" % (file_path, e))
