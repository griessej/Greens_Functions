"""
Generate a matrix with a given number of rows and columns. 
On the main diagonal are ones and one specified row or column has values unequal to zero. 
Save the full as well as the splitted matrix to file.

"""
import sys
import numpy as np 
import os 
import argparse
from scipy.sparse import csr_matrix, identity, save_npz, diags, eye, hstack, vstack 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_cols_rows", help="Number of columns and rows of the matrix.", type=int, default=1000)
    parser.add_argument("non_zero_value", help="The value in the non-zero row or column.", type=int, default=2)
    parser.add_argument("row_col_index", help="Index of row or column to be modified.", type=int, default=1)
    parser.add_argument("--row", help="Change a row to nonzero values instead of a column.", type=bool, default=False)
    args = parser.parse_args()

    # The full matrix is composed of a diagonal matrix until the non-zero col/row, a dense block for the non-zero block and 
    # a diagonal matrix to achieve a quadratic matrix
    # First diagonal block
    if (args.row == True):
        print("Change row {:d} to nonzero values.".format(args.row_col_index))
        # First matrix
        matrix_1 = eye(args.row_col_index, args.n_cols_rows, k=0, format="csr")
        # Second matrix 
        row = np.repeat(args.non_zero_value, args.n_cols_rows)
        row[args.row_col_index] = 1
        matrix_2 = csr_matrix(row)
        # Third matrix 
        matrix_3 = eye((args.n_cols_rows-args.row_col_index-1), args.n_cols_rows, k=(args.row_col_index+1), format="csr")
        # Stack along axis 0 
        matrix = vstack((matrix_1, matrix_2, matrix_3), format="csr")

    else:
        print("Change column {:d} to nonzero values.".format(args.row_col_index))
        # First matrix 
        matrix_1 = eye(args.n_cols_rows, args.row_col_index, k=0, format="csr")
        # Second matrix 
        column = np.repeat(args.non_zero_value, args.n_cols_rows)
        column[args.row_col_index] = 1
        matrix_2 = csr_matrix(column.reshape(-1,1))
        # Third matrix 
        matrix_3 = eye(args.n_cols_rows, (args.n_cols_rows-args.row_col_index-1),k=-(args.row_col_index+1), format="csr")
        # Stack along axis 1 
        matrix = hstack((matrix_1, matrix_2, matrix_3), format="csr")
    print(matrix.toarray())

    print("Format of identity matrix: ", matrix.getformat())
    print("Nonzero entries: ", matrix.count_nonzero())
    print("Density (number of nonzeros/n_col_rows^2): ", matrix.count_nonzero()/args.n_cols_rows**2)

    # Save the full matrix
    directory = "single_col_row_matrix_nonZeroValue" + str(args.non_zero_value) + "_ncols" + str(args.n_cols_rows) + "_full"
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
    directory = "single_col_row_matrix_nonZeroValue" + str(args.non_zero_value) + "_ncols" + str(args.n_cols_rows) +  "_splits_2"
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
    directory = "single_col_row_matrix_nonZeroValue" + str(args.non_zero_value) + "_ncols" + str(args.n_cols_rows) +  "_splits_4"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for rank, (start, end) in enumerate(ranges):
        print("rank {:d}: rows {:d} -- {:d}".format(rank, start, end))
        indptr_slice = matrix.indptr[start:end+1] - matrix.indptr[start]
        indices_slice = matrix.indices[matrix.indptr[start]:matrix.indptr[end]]
        data_slice = matrix.data[matrix.indptr[start]:matrix.indptr[end]]
        csr_slice = csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end - start, num_rows))
        save_npz(directory + "/csr_slice_rows_{:d}_to_{:d}.npz".format(start, end), csr_slice)
