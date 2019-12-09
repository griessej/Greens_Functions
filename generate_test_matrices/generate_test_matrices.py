"""
Generate test matrices and 

"""
import sys
import numpy as np 
import os 
from scipy.sparse import csr_matrix, identity, save_npz, random
    
if __name__ == "__main__":
    n_col_rows = 1000
    I = identity(n_col_rows, format="csr")
    print("Format of identity matrix: ", I.getformat())
    print("Nonzero entries: ", I.count_nonzero())
    print("Density (number of nonzeros/n_col_rows^2): ", I.count_nonzero()/n_col_rows**2)

    # Save the full matrix
    directory = "full_matrix"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_npz(directory + "/csr_slice_rows_0_to_{:d}.npz".format(n_col_rows), I)

    #  Two splits 
    ranges = np.array([[0, 500],[500,1000]])
    num_rows = ranges[-1, 1]
    print("Number of rows: ", num_rows)
    directory = "split_matrix_2_parts"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for rank, (start, end) in enumerate(ranges):
        print("rank {:d}: rows {:d} -- {:d}".format(rank, start, end))
        indptr_slice = I.indptr[start:end+1] - I.indptr[start]
        indices_slice = I.indices[I.indptr[start]:I.indptr[end]]
        data_slice = I.data[I.indptr[start]:I.indptr[end]]
        csr_slice = csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end - start, num_rows))
        save_npz(directory + "/csr_slice_rows_{:d}_to_{:d}.npz".format(start, end), csr_slice)

    # Four splits 
    ranges = np.array([[0, 250], [250, 500], [500, 750], [750,1000]])
    num_rows = ranges[-1, 1]
    print("Number of rows: ", num_rows)
    directory = "split_matrix_4_parts"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for rank, (start, end) in enumerate(ranges):
        print("rank {:d}: rows {:d} -- {:d}".format(rank, start, end))
        indptr_slice = I.indptr[start:end+1] - I.indptr[start]
        indices_slice = I.indices[I.indptr[start]:I.indptr[end]]
        data_slice = I.data[I.indptr[start]:I.indptr[end]]
        csr_slice = csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end - start, num_rows))
        save_npz(directory + "/csr_slice_rows_{:d}_to_{:d}.npz".format(start, end), csr_slice)

    # Generate a test matrix with density=0.01 and size 100
    sparse_test_N100 = random(100, 100, density=0.01, format="csr")
    directory = "sparse_matrix_density0.01_size100_split1"
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_npz(directory + "/csr_slice_rows_0_to_{:d}.npz".format(100), sparse_test_N100)
