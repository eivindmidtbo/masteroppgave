# masteroppgave

Brukt for masteroppgaven IT3920 vår 2024 til Eivind Midtbø Øyulvstad og Thomas Nitsche. Mye er likt fra dette repoet https://github.com/bjorafla/master , og vårt repo er i praksis en fork.

# Setup

- Install GCC
- Clone repo
- CD into traj-dist-master
- Run python setup.py install
- Run pip install .
- install other eventual dependencies (numpy, ++)

# Troubleshooting and issues

- Fixed traj-dist package issues by following: https://github.com/bguillouet/traj-dist/issues/28
  - Forced integer division in frechet.pyx as shown in: https://stackoverflow.com/questions/64932145/cython-compile-error-cannot-assign-type-double-to-int-using-mingw64-in-win

## Finding the Project Root Dynamically

You can use the following Python code to dynamically locate the `masteroppgave/root` folder:

```python
import os
import sys

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

# Example usage
project_root = find_project_root("masteroppgave")

if project_root:
    sys.path.append(project_root)
    print(f"Project root found: {project_root}")
else:
    raise RuntimeError("Could not find 'masteroppgave' directory")
```

### Folder structure

| **Folder Name**    | **Description**                             |
| ------------------ | ------------------------------------------- |
| `computation`      | Contains all computational processes.       |
| `correlation`      | Stores correlation figures.                 |
| `dataset`          | Includes all datasets and generated hashes. |
| `results_hashed`   | Results of computations on hashed data.     |
| `results_true`     | Results of computations on true data.       |
| `schemes`          | Contains LSH schemes.                       |
| `traj-dist-master` | External trajectory similarity library.     |
| `utils`            | Helper functions.                           |
| `visualization`    | Includes visualizations.                    |


### Often used files

#### Analysis

| **File Name** | **Type** | **Description** |
|---------------|----------|-----------------|
| [`bucketing_compute_disk_similarity_values_hybrid.ipynb`](computation/similarity_values/hashed/bucketing_compute_disk_similarity_values_hybrid.ipynb) | Analysis | Notebook for analysis on the hybrid version for disk scheme |
| [`bucketing_compute_disk_similarity_values.ipynb`](computation/similarity_values/hashed/bucketing_compute_disk_similarity_values.ipynb) | Analysis | Notebook for analysis on the disk scheme |
| [`bucketing_compute_grid_similarity_value_hybrid.ipynb`](computation/similarity_values/hashed/bucketing_compute_grid_similarity_values_hybrid.ipynb) | Analysis | Notebook for analysis on the hybrid version for grid scheme |
| [`bucketing_compute_grid_similarity_values.ipynb`](computation/similarity_values/hashed/bucketing_compute_grid_similarity_values.ipynb) | Analysis | Notebook for analysis on the grid scheme |

#### Runtime

| **File Name** | **Type** | **Description** |
|---------------|----------|-----------------|
| [`compute_hashed_bucketing_runtimes.ipynb`](computation/runtimes/hashed/compute_hashed_bucketing_runtimes.ipynb) | RUNTIME | Notebook for measuring runtime for different schemes with bucketing |
| [`compute_hashed_runtimes.ipynb`](computation/runtimes/hashed/compute_hashed_runtimes.ipynb) | RUNTIME | Notebook for measuring runtime for different schemes |
| [`compute_true_runtimes.ipynb`](computation/runtimes/true/compute_true_runtimes.ipynb) | RUNTIME | Notebook for measuring runtime for similarity computation on TRUE trajectories |
| [`measure_similarities.py`](utils/helpers/measure_similarities.py) | RUNTIME/HELPER | Sheet containing functions for measuring runtime |

### Helper Files
| **File Name** | **Type** | **Description** |
|---------------|----------|-----------------|
| [`lsh_bucketing.py`](schemes/lsh_bucketing.py) | Helper | Helper functions for bucketing logic |
| [`bucket_evaluation.py`](utils/helpers/bucket_evaluation.py) | Helper | Helper functions for bucketing analysis |
| [`similarity.py`](computation/similarity.py) | Helper | Sheet containing wrapper functions for analysis |