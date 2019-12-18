# Greens_Functions
# How to compile PETSc and PETSc4py
Clone the release version with all current patches by using git.

`git clone https://gitlab.com/petsc/petsc.git`

At the moment the function MatMumpsGetInverse() is wrong therefore checkout the following branch.

`git checkout hzhang/fix-mumps-GetInverse`

Load the following modules. All necessary packages are loaded from the mpi4py modules compiled by LP. It is highly recommended to use it, although there are already newer versions of the compiler and openmpi available. This prevents any errors in the compilation of petsc4py. If a different openmpi version is used for the mpi4py compilation than for the compilation of petsc, the program is executed m-fold serial( m=number of processors) instead of parallel. 
 
`module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles/`

`module load mpi4py/3.0.0-python-3.6.5-openmpi-3.1-gnu-7.3 `

# Workflow
* Generate the matrix in scipy.sparse.csr_matrix format, split the matrix and save splitted files to drive.
* Convert the matrices to PETSc binary format und save. 
* Read the matrix in a c-code and compute part of the inverse.

# Generate test matrices 
We consider the easiest excample of a matrix inversion, namely the identity matrix I=diag[1]. The size of the matrix is dim=[1000,1000] and use 1,2 and 4 cores. 
