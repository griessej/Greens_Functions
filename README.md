# Greens_Functions
# How to compile PETSc and PETSc4py
Clone the release version with all current patches by using git.

`git clone https://gitlab.com/petsc/petsc.git`

At the moment the function MatMumpsGetInverse() is wrong therefore checkout the following branch.

`git checkout hzhang/fix-mumps-GetInverse`

Load the following modules. All necessary packages are loaded from the mpi4py modules compiled by LP. It is highly recommended to use it, although there are already newer versions of the compiler and openmpi available. This prevents any errors in the compilation of petsc4py. If a different openmpi version is used for the mpi4py compilation than for the compilation of petsc, the program is executed m-fold serial( m=number of processors) instead of parallel. 
 
`module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles/`

`module load mpi4py/3.0.0-python-3.6.5-openmpi-3.1-gnu-7.3 `

We compile PETSc using the following options and MUMPS. I recommend compiling PETSc twice. For debugging purposes with debugging=1 and if using for processing data with debugging=0. This increases performance by a factor of two to three. scalapack and the fortran compiler are needed for MUMPS. Compilation of fblaslapack turned out to be necessary since it was not correctly configred on NEMO and errors occur during tests.

`./configure --download-mumps --with-shared-libraries=True --download-scalapack --download-fblaslapack --COPTFLAGS='-march=native -O2' --CXXOPTFLAGS='-marc
h=native -O2' --FOPTFLAGS=-march='native -O2' `

After the configure stage define set the variables PETSC_DIR and PETSC_ARCH in your .bashrc script.
You can either use pip to install the petsc4py version using 

`pip install petsc4py`

or you can simply clone the repository and 

`git clone https://bitbucket.org/petsc/petsc4py.git`

and run 

`python setup.py build`

`python setup.py install`

Set the pythonpath either to the .local enviroment in your home folder or to the virtual environment.

# Workflow
* Generate the matrix in scipy.sparse.csr_matrix format, split the matrix and save splitted files to drive.
* Convert the matrices to PETSc binary format und save. 
* Read the matrix in a c-code and compute part of the inverse.

# Generate test matrices 
We consider the easiest excample of a matrix inversion, namely the identity matrix I=diag[1]. The size of the matrix is dim=[1000,1000] and use 1,2 and 4 cores. 
