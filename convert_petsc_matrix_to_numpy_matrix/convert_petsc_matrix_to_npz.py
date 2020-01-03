"""
Read a matrix in PETSc binary (matrix type aij) format and convert it to scipy.sparse.npz format.  

Parameters
----------
petsc_matrix: petsc binary matrix 
    Path to the petsc matrix file in a binary format. 

output_filename: str
    Name of the .npz output file.

displ: bool
    If set to True print the petsc matrix and the converted matrix to terminal. Should be only used for small matrices!!

Output
----------
scipy.sparse.npz file: 
    scipy.sparse.npz file that contains the converted matrix. 

Restrictions
----------
None 

Example usage
----------
    python3 convert_petsc_matrix_to_npz.py ../compute_inverse/code_inverse_matmumpsgetinverse_sparse_rhs/inverse_main_off_diagonal_matrix_offValue2_ncols10 inverse_main_off_diagonal_matrix_offValue2_ncols10 --displ True

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
    parser.add_argument("petsc_matrix", help="Path to the PETSc matrix.")
    parser.add_argument("output_filename", help="Name of the .npz output file.")
    parser.add_argument("--displ", default=False, help="Print the loaded matrix to screen.", type=bool)
    args = parser.parse_args()

    """
    Load the matrix.
    """
    viewer = PETSc.Viewer().createBinary(args.petsc_matrix, "r")
    petsc_inverse_matrix = PETSc.Mat().load(viewer) 

    # Print the matrix to the screen
    if (args.displ != False):
        print("PETSc matrix: ")
        petsc_inverse_matrix.view()
    
    """
    Convert the PETSc aij matrix to a scipy.sparse.csr_matrix.
    """
    # Extract data, index pointer and indices from the PETSc matrix 
    indptr, indices, data = petsc_inverse_matrix.getValuesCSR()
    # Convert 
    csr_inverse_matrix = scipy.sparse.csr_matrix((data, indices, indptr))

    # Print the matrix to the screen
    if (args.displ != False):
        print("\nScipy.sparse.csr_matrix: ")
        print(csr_inverse_matrix)
    
    # Save it as a binary array 
    scipy.sparse.save_npz(args.output_filename, csr_inverse_matrix)


