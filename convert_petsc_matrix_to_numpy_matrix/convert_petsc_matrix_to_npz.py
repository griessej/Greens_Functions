"""
Read a 

Parameters
----------
dynamical_matrix_files: scipy.sparse.npz files
	Path to the splitted scipy.sparse.npz files.

dynamical_matrix_dimension: int
	Dimension of the full dynamical matrix.

output_filename: str
    Name of the binary output file. 

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
    parser.add_argument("petsc_matrix", help="Load this PETSc matrix and convert it to a numpy array.")
    parser.add_argument("output_filename", help="Name of the .npy output file.")
    parser.add_arguement("--displ", default=False, help="Print the loaded matrix to screen", type=bool)
    args = parser.parse_args()

    """
    Load the matrix.
    """
    viewer = PETSc.Viewer().createBinary(args.petsc_matrix, "r")
    petsc_matrix = PETSc.Mat().load(viewer) 

    # Print the matrix to the screen
    if (args.displ != False):
        petsc_matrix.view()

    # Save it as a binary array 
    np.save(args.output_filename, petsc_matrix)


