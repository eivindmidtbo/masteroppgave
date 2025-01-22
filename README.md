# masteroppgave

Bruk for masteroppgaven IT3920 vår 2024 til Joon-Erik Sæther og Endre Moseholm Gilje-Sørnes. Mye er likt fra dette repoet https://github.com/bjorafla/master , og vårt repo er i praksis en fork.

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


### Mapper og filer

#### Dataset: 
Inneholder alt dataset relevant. Metafiler og test filer for trajectories for både porto og rome. 

#### proccesing: 
Scripts for å rense og lage datasettene. 
#### visualization: 
Inneholder alt av visualisering av data.




## Finding the Project Root Dynamically

You can use the following Python code to dynamically locate the `masteroppgave/root` folder:

```python
import os
import sys

def find_project_root(target_folder="masteroppgave"):
    """Find the absolute path of a folder by searching upward."""
    currentdir = os.path.abspath(__file__)  # Get absolute script path
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