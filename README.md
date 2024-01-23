# masteroppgave

Repo for masteroppgave

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
