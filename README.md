# Greens_Functions
# How to compile PETSc and PETSc4py


We consider the easiest excample of a matrix inversion, namely the identity matrix I=diag[1]. The size of the matrix is dim=[1000,1000] and use 1,2 and 4 cores. 

# Workflow
* Generate the matrix in scipy.sparse.csr_matrix format, split the matrix and save splitted files to drive.
* Convert the matrices to PETSc binary format und save. 
* Read the matrix in a c-code and compute part of the inverse.
