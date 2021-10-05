"""
Generate a matrix which has ones on the main diagonal and negative factor values on the first off-digaonal.

"""
import sys
import numpy as np 
import os 
import argparse
from scipy.sparse import csr_matrix, identity, save_npz, diags
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_cols_rows", help="Number of columns and rows of the test matrix.", type=int, default=1000)
    parser.add_argument("off_diagonal_value", help="Value on the first off_diagonal.", type=float, default=1)
    args = parser.parse_args()

    off_diagonal = -np.repeat(args.off_diagonal_value, args.n_cols_rows)
    main_diagonal = np.repeat(1, args.n_cols_rows)
    diagonals = np.stack((main_diagonal, off_diagonal))
    matrix = diags(diagonals, offsets=[0,1], format="csr")
    print("Format of identity matrix: ", matrix.getformat())
    print("Nonzero entries: ", matrix.count_nonzero())
    print("Density (number of nonzeros/n_col_rows^2): ", matrix.count_nonzero()/args.n_cols_rows**2)
    print("Print matrix, turn off for larger matrices\n: ", matrix)
    
    # Save the full matrix
    directory = "main_off_diagonal_matrix_offValue" + str(args.off_diagonal_value) + "_ncols" + str(args.n_cols_rows) + "_full"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_npz(directory + "/csr_slice_rows_0_to_{:d}.npz".format(args.n_cols_rows), matrix)

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
    directory = "main_off_diagonal_matrix_offValue" + str(args.off_diagonal_value) + "_ncols" + str(args.n_cols_rows) +  "_splits_2"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for rank, (start, end) in enumerate(ranges):
        print("rank {:d}: rows {:d} -- {:d}".format(rank, start, end))
        indptr_slice = matrix.indptr[start:end+1] - matrix.indptr[start]
        indices_slice = matrix.indices[matrix.indptr[start]:matrix.indptr[end]]
        data_slice = matrix.data[matrix.indptr[start]:matrix.indptr[end]]
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
    directory = "main_off_diagonal_matrix_offValue" + str(args.off_diagonal_value) + "_ncols" + str(args.n_cols_rows) +  "_splits_4"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for rank, (start, end) in enumerate(ranges):
        print("rank {:d}: rows {:d} -- {:d}".format(rank, start, end))
        indptr_slice = matrix.indptr[start:end+1] - matrix.indptr[start]
        indices_slice = matrix.indices[matrix.indptr[start]:matrix.indptr[end]]
        data_slice = matrix.data[matrix.indptr[start]:matrix.indptr[end]]
        csr_slice = csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end - start, num_rows))
        save_npz(directory + "/csr_slice_rows_{:d}_to_{:d}.npz".format(start, end), csr_slice)
