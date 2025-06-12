# Masteroppgave

Used for the master's thesis IT3920, spring 2025, by Eivind Midtbø Øyulvstad and Thomas Nitsche. Much is the same as in this repo: https://github.com/bjorafla/master, and our repo is essentially a fork.

In our thesis paper, the bucketing strategy referred to as *strict* is called *original* in the codebase.





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


### Folder structure

| **Folder Name**    | **Description**                             |
| ------------------ | ------------------------------------------- |
| `computation`      | Contains all computational processes.       |
| `dataset`          | Includes all datasets and generated hashes. |
| `results_hashed`   | Results of computations on hashed data.     |
| `results_analysis`   | Analysis **notebooks**.     |
| `results_true`     | Results of computations on true data.       |
| `schemes`          | Contains LSH schemes.                       |
| `traj-dist-master` | External trajectory similarity library.     |
| `utils`            | Helper functions.                           |
| `visualization`    | Includes visualizations.                    |

### Often used files

#### Analysis

| **File Name**                                                                                                                                         | **Type** | **Description**                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------- |
| [`bucketing_compute_disk_similarity_values_hybrid.ipynb`](computation/similarity_values/hashed/bucketing_compute_disk_similarity_values_hybrid.ipynb) | Analysis | Notebook for analysis on the hybrid version for disk scheme |
| [`bucketing_compute_disk_similarity_values.ipynb`](computation/similarity_values/hashed/bucketing_compute_disk_similarity_values.ipynb)               | Analysis | Notebook for analysis on the disk scheme                    |
| [`bucketing_compute_grid_similarity_values_hybrid.ipynb`](computation/similarity_values/hashed/bucketing_compute_grid_similarity_values_hybrid.ipynb) | Analysis | Notebook for analysis on the hybrid version for grid scheme |
| [`bucketing_compute_grid_similarity_values.ipynb`](computation/similarity_values/hashed/bucketing_compute_grid_similarity_values.ipynb)               | Analysis | Notebook for analysis on the grid scheme                    |
| [`bucketing_correlation_disk.ipynb`](result_analysis/correlation/bucketing_correlation_disk.ipynb)                                                    | Analysis/Correlation | Notebook for correlation analysis on the disk scheme        |
| [`bucketing_correlation_grid.ipynb`](result_analysis/correlation/bucketing_correlation_grid.ipynb)                                                    | Analysis/Correlation | Notebook for correlation analysis on the grid scheme        |

#### Runtime

| **File Name**                                                                                                    | **Type**       | **Description**                                                                |
| ---------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------ |
| [`compute_hashed_bucketing_runtimes_hybrid.ipynb`](computation/runtimes/hashed/compute_hashed_bucketing_runtimes_hybrid.ipynb) | RUNTIME        | Notebook for measuring runtime for hybrid version of different schemes with bucketing            |
| [`compute_hashed_bucketing_runtimes.ipynb`](computation/runtimes/hashed/compute_hashed_bucketing_runtimes.ipynb) | RUNTIME        | Notebook for measuring runtime for different schemes with bucketing            |
| [`compute_hashed_runtimes.ipynb`](computation/runtimes/hashed/compute_hashed_runtimes.ipynb)                     | RUNTIME        | Notebook for measuring runtime for different schemes                           |
| [`compute_true_runtimes.ipynb`](computation/runtimes/true/compute_true_runtimes.ipynb)                           | RUNTIME        | Notebook for measuring runtime for similarity computation on TRUE trajectories |
| [`measure_similarities.py`](utils/helpers/measure_similarities.py)                                               | RUNTIME/HELPER | Sheet containing functions for measuring runtime                               |

### Helper Files

| **File Name**                                                | **Type** | **Description**                                 |
| ------------------------------------------------------------ | -------- | ----------------------------------------------- |
| [`lsh_bucketing.py`](schemes/lsh_bucketing.py)               | Helper   | Helper functions for bucketing logic            |
| [`bucket_evaluation.py`](utils/helpers/bucket_evaluation.py) | Helper   | Helper functions for bucketing analysis         |
| [`similarity.py`](computation/similarity.py)                 | Helper   | Sheet containing wrapper functions for analysis |
| [`disk_correlation_bucketing.py`](result_analysis/disk_correlation_bucketing.py)| Helper   | Sheet containing correlation functions for disk scheme |
| [`grid_correlation_bucketing.py`](result_analysis/grid_correlation_bucketing.py)| Helper   | Sheet containing correlation functions for grid scheme |
