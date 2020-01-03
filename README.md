# Greens_Functions
# How to compile PETSc and PETSc4py
Clone the release version of PETSc with all current patches by using

`git clone -b maint https://gitlab.com/petsc/petsc.git petsc`

Use 

`git pull `

in the petsc directory anytime to obtain new patches that have been added since your git clone or last git pull.
Load the following modules. All necessary packages e.g compilers, mpi,... are loaded from the mpi4py module compiled by LP. It is highly recommended to use it, although there are already newer versions of the compiler and openmpi available. This prevents any errors in the compilation of petsc4py. If a different openmpi version is used for the compilation of mpi4py than for the compilation of petsc, the program is executed m-fold serial(m=number of processors) instead of parallel. Therefore run the following commands: 
 
`module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles/`

`module load mpi4py/3.0.0-python-3.6.5-openmpi-3.1-gnu-7.3 `

Change to the PETSc directory and use the following command to configure

`./configure --download-mumps --with-shared-libraries=True --download-scalapack --download-fblaslapack --COPTFLAGS='-march=native -O2' --CXXOPTFLAGS='-march=native -O2' --FOPTFLAGS=-march='native -O2 --with-debugging=0 `

For debugging and testing purposes the flag --with-debugging should be set to 1. Turning the debugging flag off increases the performance by a factor of two to three. For production runs i highly recommend truning it off! The fortan compilers and scalapack are needed for the compilation of MUMPS. MUMPS is necessary in order to use parallel direct solvers and therefore perform the LU-factorization in parallel. Furthermore it is needed to take advantage of the MUMPSGetInverse() function. This allows a parallel computation of a part of the inverse of a matrix. Compilation of fblaslapack turned out to be necessary since it was not correctly configred on NEMO and errors occur during tests. After the configure, make and test stage set the variables PETSC_DIR and PETSC_ARCH in your .bashrc script and make sure you are loading these paths when running on NEMO. Example:

`export PETSC_DIR=/home/fr/fr_fr/fr_jg1080/Libaries/petsc_3.12.0_debugging_0`

`export PETSC_ARCH=arch-linux-c-opt` 

We use a virtual python environment in order to set up the petsc4py version. Therefore we run 

`python3 -m venv venv_petsc4py_slepc4py`

and activate the virtual enviromnent using 

`source bin/activate`

and install PETSc4Py using pip

`pip3 install petsc4py`

In summary the following paths and exports are necessary 

`module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles/`

`module load mpi4py/3.0.0-python-3.6.5-openmpi-3.1-gnu-7.3`

`export PETSC_DIR=/work/ws/nemo/fr_wn1007-2019-09-23_HEA_greens_function-0/Tools/petsc_3.12.0_debugging_0`

`export PETSC_ARCH=arch-linux-c-opt`

`source /work/ws/nemo/fr_wn1007-2019-09-23_HEA_greens_function-0/Tools/venv_petsc4py-3.12.0_debugging_0/bin/activate`

`export PYTHONPATH="/work/ws/nemo/fr_wn1007-2019-09-23_HEA_greens_function-0/Tools/venv_petsc4py-3.12.0_debugging_0/lib/python3.6/site-packages/:$PYTHONPATH"`


# Workflow
* Generate the matrix in scipy.sparse.csr_matrix format, split the matrix and save splitted files to drive.
* Convert the matrices to PETSc binary format und save. 
* Read the matrix in a c-code and compute part of the inverse.

# Generate test matrices 
We consider the easiest excample of a matrix inversion, namely the identity matrix I=diag[1]. The size of the matrix is dim=[1000,1000] and use 1,2 and 4 cores. 
