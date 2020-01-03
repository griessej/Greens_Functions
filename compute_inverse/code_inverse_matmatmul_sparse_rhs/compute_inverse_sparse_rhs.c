static char help[] ="Compute a part of the inverse of a sparse matrix. This code requires that PETSc was configured with MUMPS since we are dealing with large matrices \
		     and therefore use a parallel LU factorization. We compute the inverse by solving the equation A*X=RHS. Where A is our Matrix, X is the inverse and RHS is the identity matrix.\
		     Note that the number of columns nrhs of X can be chosen smaller than the number of columns N in A. Therefore only a part of the inverse is computed in X. \n \
		     In this code we use a sparse representation of the RHS matrix in MUMPS. \n \
		     Input parameters include\n\
  			-fin <input_file> : file to load \n \
                	-fout <input_file> : file to load \n \
			-nrhs <numberofcolumns> : Number of columns to be compute \n \
                        -displ <Bool>: Print matrices to terminal \n\
		     Example usage: \n \
		         mpiexec -np 2 ./compute_inverse_sparse_rhs -fin ../../convert_to_binary_petsc_matrix/identity_matrix_prefactor3_ncols10 -nrhs 5 -displ";

#include <stdio.h>
#include <petscmat.h>
#include <petscviewer.h> 

int main(int argc, char **args){
    PetscErrorCode 	ierr; 					// Datatype used for return error code
    PetscMPIInt		size,rank; 				// Datatype used to represent 'int' parameters to MPI functions.
#if defined(PETSC_HAVE_MUMPS)
    Mat			A,F,spRHST,X;				// Abstract PETSc matrix object used to manage all linear operators in PETSc
    PetscViewer		fd; 					// Abstract PETSc object that helps view (in ASCII, binary, graphically etc) other PETSc objects
    PetscBool      	flg1,flg2;				// Logical variable. Actually an int in C.
    PetscBool		displ=PETSC_FALSE;			// Display matrices 
    PetscInt		M,N,m,n,rstart,rend,nrhs,i;		// PETSc type that represents an integer, used primarily to represent size of arrays and indexing into arrays.
    PetscScalar         v;                              	// PETSc type that represents either a double precision real number,...
    char		inputfile[1][PETSC_MAX_PATH_LEN]; 	// Input file name 
//    char		outputfile[1][PETSC_MAX_PATH_LEN]; 	// Outputfile file name 
#endif

    // Initializes PETSc and MPI. Get size and rank of MPI.
    ierr = PetscInitialize(&argc, &args, (char*)0, help);if (ierr){return ierr;}
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

    //Check if PETSc was configured with MUMPS. IF not print error message and exit 
#if !defined(PETSC_HAVE_MUMPS)
    if (!=rank){ierr = PetscPrintf(PETSC_COMM_SELF, "This code requires MUMPS, exit...\n");CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
    }
#else

    // Check if displ is set. If True the matrices are printed to the terminal
    ierr = PetscOptionsGetBool(NULL, NULL, "-displ", &displ, NULL);CHKERRQ(ierr);

    // Load matrix A from file 
    ierr = PetscOptionsGetString(NULL, NULL, "-fin" ,inputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Load matrix in: %s \n", inputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, inputfile[0], FILE_MODE_READ, &fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);  
    ierr = MatLoad(A, fd);CHKERRQ(ierr);
    // Print matrix A 
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix A from file:\n", nrhs);
        ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }
    // Check if matrix is quadratic
    ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
    if (M != N){
        //Macro that is called when an error has been detected.
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Detected a rectangular matrix: (%d, %d)", M, N);
    }
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for Matrix A, rank:  %i, size: %i, rstart: %i, rend: %i \n", rank, size, rstart, rend);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);

    // Set the number of columns of the inverse to be computed. 
    nrhs = N;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, &flg2);CHKERRQ(ierr);
    
    // Set up dense matrix X which holds the solution.
    ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
    ierr = MatSetSizes(X,m,PETSC_DECIDE,PETSC_DECIDE,nrhs);CHKERRQ(ierr);
    ierr = MatSetType(X,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(X);CHKERRQ(ierr);
    ierr = MatSetUp(X);CHKERRQ(ierr);

    ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
    // Create SpRHST for inv(A) with sparse RHS stored in the host.
    // PETSc does not support compressed column format which is required by MUMPS for sparse RHS matrix,
    // thus user must create spRHST=spRHS^T and call MatMatTransposeSolve()
    // User must create B^T in sparse compressed row format on the host processor and call MatMatTransposeSolve() to implement MUMPS' MatMatSolve().
    ierr = MatCreate(PETSC_COMM_WORLD, &spRHST);CHKERRQ(ierr);
    if (!rank){
        ierr = MatSetSizes(spRHST,nrhs,M,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    else{
        ierr = MatSetSizes(spRHST,0,0,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = MatSetType(spRHST,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(spRHST);CHKERRQ(ierr);
    ierr = MatSetUp(spRHST);CHKERRQ(ierr);
    if (!rank){
        v = 1.0;
        for(i=0;i<nrhs;i++){
            ierr = MatSetValues(spRHST,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(spRHST,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(spRHST, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // Print information
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCompute %i columns of the inverse using LU-factorization in MUMPS!\n", nrhs);

    // Factorize the Matrix using a parallel LU factorization in MUMPS
    ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &F);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(F, A, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(F, A, NULL);CHKERRQ(ierr);

    // Solves AX = B^T
    ierr = MatMatTransposeSolve(F,spRHST,X);CHKERRQ(ierr);

    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"First %D columns of inv(A) with sparse RHS:\n", nrhs);
        ierr = MatView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);
    }
    
    // Free data structures
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr);
    ierr = MatDestroy(&spRHST);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
    
#endif
}


