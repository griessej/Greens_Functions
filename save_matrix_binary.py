"""
Read and assemble a Petsc matrix in parallel. Save it in order to read the file in c

Parameters
----------
dynamical_matrix_files: scipy.sparse.npz files
	Path to the splitted scipy.sparse.npz files.

dynamical_matrix_dimension: int
	Dimension of the full dynamical matrix.


Output
----------
	

Restrictions
----------


"""
import sys
import numpy as np 
import argparse
import scipy.sparse

# Import PETSc4py
import petsc4py 
# Initializes the PETSc database and MPI. PetscInitialize() calls MPI_Init() if that has yet to be called,
# so this routine should always be called near the beginning of your program -- usually the very first line!
petsc4py.init(sys.argv)
from petsc4py import PETSc

comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

    
if __name__ == "__main__":
    # Get options from command line 
    parser = argparse.ArgumentParser()
    parser.add_argument("dynamical_matrix_files", help="Path to the splitted dynamical matrix files.")
    parser.add_argument("dynamical_matrix_dimension", help="Dimension of the full dynamical matrix.", type=int)
    args = parser.parse_args()

    """
    Construct the dynamical matrix in a PETSc matrix format.
    """
    # Create a matrix A in parallel
    A = PETSc.Mat().create(comm=comm)
    A.setSizes((args.dynamical_matrix_dimension, args.dynamical_matrix_dimension))
    A.setFromOptions()
    # If the user has not set preallocation for this matrix then a default preallocation that is likely to be inefficient is used.
    # If a suitable preallocation routine is used, this function does not need to be called.
    A.setUp()
    # Currently, all PETSc parallel matrix formats are partioned by contiguous chunks of rows across the processors. Determine which 
    # rows of the matrix are locally owned 
    Rstart, Rend = A.getOwnershipRange()

    # Print information
    print("rank, size, start_frame, end_frame \n", rank, " / ", size, " / ", Rstart, " / ", Rend)

    # Load the corresponding part of the dynamical matrix 
    D_filename = args.dynamical_matrix_files + "/csr_slice_rows_{:d}_to_{:d}.npz".format(Rstart, Rend)
    D_sparse_mn = scipy.sparse.load_npz(D_filename)
    csr = (D_sparse_mn.indptr, D_sparse_mn.indices, D_sparse_mn.data)

    # Assemble the dynamical matrix as PETSc matrix 
    A = PETSc.Mat().createAIJ(size=(args.dynamical_matrix_dimension, args.dynamical_matrix_dimension), csr=csr, comm=comm)
    A.assemble()

    # Write a binary file in parallel 
    print("Here")
    viewer = PETSc.Viewer().createBinary("test", mode="w", comm=PETSc.COMM_WORLD)
    viewer(A)


