"""Convert a binary scipy sparse matrix file into a petsc sparse matrix file.

Output
------
output_file.bin + output_file.info: bin, txt
    The bin file contains the dynamical matrix in a PETSc binary format. The
    .info-file contains the size of the blocks to use if the matrix is read
    into a block oriented data structure.
	
Restrictions
------------
None 

Example usage
-------------
python3 save_matrix_binary_serial.py input_matrix.npz output_matrix.bin --upper_left 9 --displ True
"""

import sys
import numpy as np
import argparse
import scipy.sparse
import petsc4py

# Initializes the PETSc database and MPI. PetscInitialize() calls MPI_Init() if
# that has yet to be called, so this routine should always be called near the
# beginning of your program -- usually the very first line!
petsc4py.init(sys.argv)
from petsc4py import PETSc

if PETSc.COMM_WORLD.getSize() > 1:
    print(
        "This program is intended to one sparse.csr file in bsr matrix, convert it to csr and save as binary! Exit!"
    )
    sys.exit(2)


def main():
    # Get options from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to sparse matrix in scipy bsr format")
    parser.add_argument("output_file", help="Path to output file")
    parser.add_argument(
        "--upper_left",
        type=int,
        help="Extract square block in the upper left with the first N_u values. N_u needs to be a multiple of the block size of the sparse matrix.",
    )
    parser.add_argument(
        "--displ", action="store_true", help="Print the converted matrix to screen?"
    )
    args = parser.parse_args()
    # Load the sparse array, slice if necessary, and convert to CSR format
    scipy_mat = scipy.sparse.load_npz(args.input_file)
    print("Current format of sparse array: ", scipy_mat.getformat())
    if args.upper_left is not None:
        ul_indices = np.arange(args.upper_left, dtype=int)
        tmp = scipy_mat.tocsc()[:, ul_indices]
        scipy_mat = tmp.tocsr()[ul_indices, :]
    else:
        scipy_mat = scipy_mat.tocsr()
    print("New format of sparse matrix: ", scipy_mat.getformat())
    csr = (scipy_mat.indptr, scipy_mat.indices, scipy_mat.data)
    assert scipy_mat.shape[0] == scipy_mat.shape[1]
    mat_size = scipy_mat.shape[0]
    # Assemble the dynamical matrix as PETSc matrix
    petsc_mat = PETSc.Mat().createAIJ(
        size=(mat_size, mat_size), csr=csr, comm=PETSc.COMM_WORLD
    )
    petsc_mat.assemble()
    if args.displ:
        petsc_mat.view()
    viewer = PETSc.Viewer().createBinary(
        args.output_file, mode="w", comm=PETSc.COMM_WORLD
    )
    viewer(petsc_mat)


if __name__ == "__main__":
    main()
