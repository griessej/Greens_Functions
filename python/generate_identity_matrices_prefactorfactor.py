"""
Generate an identity matrix with a given number of rows and columns. Save the full as well as the splitted identity matrix to file.

"""
import sys
import numpy as np 
import os 
import argparse
from scipy.sparse import csr_matrix, identity, save_npz, diags
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_cols_rows", help="Numb of columns and rows of the test matrix.", type=int, default=1000)
    parser.add_argument("prefactor", help="Value on the diagonal of the matrix.", type=int, default=1)
    args = parser.parse_args()

    diagonals = np.repeat(args.prefactor, args.n_cols_rows)
    diagonal_matrix = diags(diagonals, offsets=0, format="csr")
    print("Format of identity matrix: ", diagonal_matrix.getformat())
    print("Nonzero entries: ", diagonal_matrix.count_nonzero())
    print("Density (number of nonzeros/n_col_rows^2): ", diagonal_matrix.count_nonzero()/args.n_cols_rows**2)

    # Save the full matrix
    directory = "identity_matrix_prefactor" + str(args.prefactor) + "_ncols" + str(args.n_cols_rows) + "_full"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_npz(directory + "/csr_slice_rows_0_to_{:d}.npz".format(args.n_cols_rows), diagonal_matrix)

    #  Two splits 
    print("----------------")
    ranges = np.zeros((2,2))
    indices = np.linspace(0, args.n_cols_rows, 3, dtype=int)
    for i in range(2):
        ranges[i,0] = indices[i]
        ranges[i,1] = indices[i+1] 
    ranges = ranges.astype(int)
    num_rows = ranges[-1, 1]
    print("Number of rows: ", num_rows)
    print("Ranges: ", ranges)
    directory = "identity_matrix_prefactor" + str(args.prefactor) + "_ncols" + str(args.n_cols_rows) +  "_splits_2"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for rank, (start, end) in enumerate(ranges):
        print("rank {:d}: rows {:d} -- {:d}".format(rank, start, end))
        indptr_slice = diagonal_matrix.indptr[start:end+1] - diagonal_matrix.indptr[start]
        indices_slice = diagonal_matrix.indices[diagonal_matrix.indptr[start]:diagonal_matrix.indptr[end]]
        data_slice = diagonal_matrix.data[diagonal_matrix.indptr[start]:diagonal_matrix.indptr[end]]
        csr_slice = csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end - start, num_rows))
        save_npz(directory + "/csr_slice_rows_{:d}_to_{:d}.npz".format(start, end), csr_slice)

    # Four splits 
    print("----------------")
    ranges = np.zeros((4,2))
    indices = np.linspace(0, args.n_cols_rows, 5)
    for i in range(4):
        ranges[i,0] = indices[i]
        ranges[i,1] = indices[i+1] 
    ranges = ranges.astype(int)
    num_rows = ranges[-1, 1]
    print("Number of rows: ", num_rows)
    print("Ranges: ", ranges)
    directory = "identity_matrix_prefactor" + str(args.prefactor) + "_ncols" + str(args.n_cols_rows) +  "_splits_4"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for rank, (start, end) in enumerate(ranges):
        print("rank {:d}: rows {:d} -- {:d}".format(rank, start, end))
        indptr_slice = diagonal_matrix.indptr[start:end+1] - diagonal_matrix.indptr[start]
        indices_slice = diagonal_matrix.indices[diagonal_matrix.indptr[start]:diagonal_matrix.indptr[end]]
        data_slice = diagonal_matrix.data[diagonal_matrix.indptr[start]:diagonal_matrix.indptr[end]]
        csr_slice = csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end - start, num_rows))
        save_npz(directory + "/csr_slice_rows_{:d}_to_{:d}.npz".format(start, end), csr_slice)
