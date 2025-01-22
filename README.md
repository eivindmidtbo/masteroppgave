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