#!/usr/bin/env python3
"""
Read and assemble a Petsc matrix in parallel. Save it in order to read the file in the c-code of PETSc to compute the inverse.
Run on one processor to read the full matrix, run on m processors to read slices of the matrix, combine them and save.

Parameters
----------
dynamical_matrix_files: scipy.sparse.npz files
    Path to the splitted scipy.sparse.npz files.

dynamical_matrix_dimension: int
    Dimension of the full dynamical matrix.

output_filename: str
    Name of the binary output file. 

--displ: bool
    Print the converted matrix to screen.    

Output
----------
output_filename.bin + output_filename.info: bin, txt
    The bin file contains the dynamical matrix in a PETSc binary format. The .info file contains the size of the blocks to use 
    if the matrix is read into a block oriented data structure.
	

Restrictions
----------
None 

Example usage
----------
python3 save_matrix_binary.py ../generate_test_matrices/main_off_diagonal_matrix_offValue2_ncols10_full/ 10 test --displ True

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
    parser.add_argument("dynamical_matrix_files", help="Path to the files that contain the splitted dynamical matrix files.")
    parser.add_argument("dynamical_matrix_dimension", help="Dimension of the full dynamical matrix. If N is the number of particles dimension = 3N.", type=int)
    parser.add_argument("output_filename", help="Name of the binary output file.")
    parser.add_argument("--displ", default=False, help="Print the converted matrix to screen.", type=bool)
    args = parser.parse_args()

    if (size == 1):
        print("This program is intended to read multiple sparse.csr slices that are converted to csr format before! Exit!")
        sys.exit(2)

    """
    Construct the matrix in a PETSc matrix format.
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
    R_start, R_end = A.getOwnershipRange()

    # Print rank and the corresponding rows 
    print("rank, size, start_row, end_row \n", rank, " / ", size, " / ", R_start, " / ", R_end)

    # Load the corresponding part of the dynamical matrix 
    D_filename = args.dynamical_matrix_files + "/csr_slice_rows_{:d}_to_{:d}.npz".format(R_start, R_end)
    D_sparse_mn = scipy.sparse.load_npz(D_filename)
    csr = (D_sparse_mn.indptr, D_sparse_mn.indices, D_sparse_mn.data)

    # Assemble the dynamical matrix as PETSc matrix 
    A = PETSc.Mat().createAIJ(size=(args.dynamical_matrix_dimension, args.dynamical_matrix_dimension), csr=csr, comm=comm)
    A.assemble()
    
    # Print the matrix to the terminal 
    if (args.displ != False):
        A.view()

    # Write a binary file in parallel 
    viewer = PETSc.Viewer().createBinary(args.output_filename, mode="w", comm=PETSc.COMM_WORLD)
    viewer(A)


