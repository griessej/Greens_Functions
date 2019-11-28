"""
Minimal working example 

"""
import sys
import numpy as np 

# Import PETSc4py
import petsc4py 
petsc4py.init(sys.argv)
from petsc4py import PETSc

comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

    
if __name__ == "__main__":
    # Variables
    n = 100

    # Create a matrix A in parallel
    A = PETSc.Mat().create(comm=comm)
    A.setSizes((n, n))
    A.setFromOptions()
    A.setUp()
    # 
    Rstart, Rend = A.getOwnershipRange()

    # Print information
    print("rank, size, start_frame, end_frame \n", rank, " / ", size, " / ", Rstart, " / ", Rend)

    # 
    for i in range(Rstart, Rend):
        if i == Rend-1:
            A[i,i] = 2
        else: 
            A[i,i] = 1

    A.assemble()

    # Write a binary file in parallel 
    viewer = PETSc.Viewer().createBinary("test", mode="w", comm=PETSc.COMM_WORLD)
    viewer(A)
    #viewer = PETSc.Viewer().createASCII("test", comm=PETSc.COMM_WORLD)
    #A.view()
