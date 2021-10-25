# Greens_Functions
# How to compile PETSc and PETSc4py

# Workflow for the inversion of test matrices
* Generate the test matrices with desired shape (number of cols and rows) and desired value for non-zero entries. The matrices are saved in the scipy.sparse.npz format. 
* Load the scipy_sparse_npz matricx and convert it to the PETSc matrix format. 
* Compute the inverse of the test matrix by either use a matmatsolve() with a dense or a sparse right-hand-side (RHS), or by using the MUMPSGetInverse Function. In order to make an executable make sure to add the PETSCDIR and PETSCARCH variables in the makefile. If you want to check timing and memory consumpation run the PETSc program with -log_view.
* Convert the PETSc matrix to a scipy.sparse.npz matrix.

# Workflow for inversion of the dynamical matrix 
* Compute the dynamical matrix in scipy.sparse.csr_matrix format. If necessary split the matrix and save splitted files (This is in general not needed!)
* Convert the matrix to the PETSc binary format. 
* Read the matrix in a c-code and compute part of the inverse.
* Convert the PETSc matrix to a scipy.sparse.npz matrix.
