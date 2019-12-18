static char help[] ="Compute a part of the inverse of a sparse matrix. This code requires that PETSc was configured with MUMPS since we are dealing with large matrices \
		     and therefore use a parallel LU factorization. We compute the inverse by using MUMPS for selceted columns of the inverse. \n \
		     Input parameters include\n\
  			-fin <input_file> : file to load \n \
                	-fout <input_file> : file to load \n \
			-nrhs <numberofcolumns> : Compute the entries of the inverse for (col,row) = 0-nrhs.\n \
                        -displ <Bool>: Print matrices to terminal \n\
		     Example usage: \n\
			mpiexec -np 2 ./compute_inverse_matmatmul -fin ../../convert_to_binary_petsc_matrix/identity_matrix_prefactor3_ncols10 -fout test -nrhs 3 -displ -log_view";

#include <stdio.h>
#include <petscmat.h>
#include <petscviewer.h> 

int main(int argc, char **args){
    PetscErrorCode 	ierr; 					// Datatype used for return error code
    PetscMPIInt		size,rank; 				// Datatype used to represent 'int' parameters to MPI functions.
#if defined(PETSC_HAVE_MUMPS)
    Mat			A,F,spRHST,X;				// Abstract PETSc matrix object used to manage all linear operators in PETSc
    Mat 		spRHS = NULL;				// 
    PetscViewer		fd; 					// Abstract PETSc object that helps view (in ASCII, binary, graphically etc) other PETSc objects
    PetscBool      	flg1,flg2;				// Logical variable. Actually an int in C.
    PetscBool		displ=PETSC_FALSE;			// Display matrices 
    PetscInt		M,N,m,n,nrhs,rstart,rend,i;		// PETSc type that represents an integer, used primarily to represent size of arrays and indexing into arrays.
    PetscScalar         v;                              	// PETSc type that represents either a double precision real number,...
    char		inputfile[1][PETSC_MAX_PATH_LEN]; 	// Input file name 
    char		outputfile[1][PETSC_MAX_PATH_LEN]; 	// Outputfile file name 
#endif

    // Initializes PETSc and MPI. 
    // help is an optional character string, which will be printed if the program is run with the -help option.
    // All PETSc routines return a PetscErrorCode, which is an integer indicating whether an error has occured during the call.
    // nonzero = Error, otherwise zero
    // By default, PetscInitialize() sets the PETSc "world" communicator to MPI_COMM_WORLD
    ierr = PetscInitialize(&argc, &args, (char*)0, help);
    if (ierr){
        return ierr;  
    }
    // Determines the size of the group associated with a communicator
    // Checks error code, if non-zero callse th error handler and then returns
    // CHKERRQ(ierr) is fundamentally a macro replacement for if (ierr) return(PetscError(...,ierr,...));
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    // Determines the rank of the calling process in the communicator
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

    //Check if PETSc was configured with MUMPS. IF not print error message and exit 
#if !defined(PETSC_HAVE_MUMPS)
    if (!=rank){
        ierr = PetscPrintf(PETSC_COMM_SELF, "This code requires MUMPS, exit...\n");
        CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
    }
#endif

    // Check if displ is set. If True the matrices are printed to the terminal
    ierr = PetscOptionsGetBool(NULL, NULL, "-displ", &displ, NULL);CHKERRQ(ierr);
    
    // Load the matrix
    // -----------------------------------------------------------------------
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Loading matrix! \n---------------\n");CHKERRQ(ierr);
    // Gets the string value for a particular option in the database.
    ierr = PetscOptionsGetString(NULL, NULL, "-fin" ,inputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr);
    // Prints to standard out, only from the first processor in the communicator.
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Load matrix in: %s \n", inputfile[0]);CHKERRQ(ierr);
    // Opens a file for binary input/output. 
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, inputfile[0], FILE_MODE_READ, &fd);CHKERRQ(ierr);
    // Creates a matrix where the type is determined from either a call to MatSetType(), the options database or by reading a file.
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    // Builds matrix object for a particular matrix type. This matrix type is identical to MATSEQAIJ when constructed with a single process communicator, and MATMPIAIJ otherwise.
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);  
    // Loads a matrix that has been stored in binary/HDF5 format with MatView().
    ierr = MatLoad(A, fd);CHKERRQ(ierr);

    // Print initial matrix to terminal
    if (displ){
        ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    
    // Returns the numbers of global rows and global columns in a matrix.
    ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
    // Check if matrix is quadratic
    if (M != N){
        //Macro that is called when an error has been detected.
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Detected a rectangular matrix: (%d, %d)", M, N);
    }
    // Returns the number of rows and columns in a matrix stored locally. m = number of local rows, n = number of local columns.
    ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
    // Returns the range of matrix rows owned by this processor, assuming that the matrix is laid out with the first n1 rows on the first processor, the next n2 rows on the second, etc.
    ierr = MatGetOwnershipRange(A,&rstart,&rend);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for Matrix A, rank:  %i, size: %i, rstart: %i, rend: %i, local row: %i, local column: %i, global row: %i, global column: %i \n", rank, size, rstart, rend, m, n, M, N);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    // -----------------------------------------------------------------------
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\nFactorization! \n---------------\n");CHKERRQ(ierr);
    // Factorize the Matrix using a parallel LU factorization in MUMPS
    // Cholesky is not possible since we are dealing with postivie semi-definite matrices --> Could be numerically unstable!
    PetscPrintf(PETSC_COMM_WORLD, "Use parallel LU factorization! \n");
    // Returns a matrix suitable to calls to MatXXFactorSymbolic().
    ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &F);CHKERRQ(ierr);
    // Performs symbolic LU factorization of matrix.
    ierr = MatLUFactorSymbolic(F, A, NULL, NULL, NULL);CHKERRQ(ierr);
    // Performs numeric LU factorization of a matrix.
    ierr = MatLUFactorNumeric(F, A, NULL);CHKERRQ(ierr);

    // -----------------------------------------------------------------------
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\nCreate sparse csr right-hand side! \n---------------\n");CHKERRQ(ierr);
    // Gets the integer value for a particular option in the database.
    // Set the number of colums on the right hand side
    nrhs = N;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, &flg2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Compute %i columns of the inverse\n ", nrhs);
    
    // Compute inv(A) with sparse right hand side (spRHS) stored in the host.
    // spRHST = [e[0],...,e[nrhs-1]]^T, dense X holds first nrhs columns of inv(A) 
    // PETSc does not support compressed column format which is required by MUMPS for sparse RHS matrix,
    // thus user must create spRHST=spRHS^T and call MatMatTransposeSolve().
    // Generate a sparse right hand side 

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank:  %i, size: %i, global row: %i, global column: %i \n", rank, size, M, N);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    ierr = MatCreate(PETSC_COMM_WORLD, &spRHST);CHKERRQ(ierr);
    // MUMPS requires the sparse right hand side (RHS) to be centralized on the host
    if (!rank){
        // Keep in mind that this is the transpose RHS!
        ierr = MatSetSizes(spRHST, N, M, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
    } 
    else{
        // Every other rank has a (len(0),len(0))-dimensional matrix 
        ierr = MatSetSizes(spRHST, 0, 0, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
    }
    // Builds matrix object for a particular matrix type.
    // his matrix type is identical to MATSEQAIJ when constructed with a single process communicator, and MATMPIAIJ otherwise.
    ierr = MatSetType(spRHST, MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(spRHST);CHKERRQ(ierr);
    ierr = MatSetUp(spRHST);CHKERRQ(ierr);
    // Set ones on the diagonal on rank 0
    if (!rank){
        v = 1.0;
        for (i=0; i<nrhs;i++){
            ierr = MatSetValues(spRHST, 1, &i, 1, &i, &v, INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    // Begin and finalize assembling the matrix. 
    ierr = MatAssemblyBegin(spRHST, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(spRHST, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nspRHST:\n");CHKERRQ(ierr);
        ierr = MatView(spRHST, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    ierr = MatGetSize(spRHST, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(spRHST, &m, &n);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank:  %i, size: %i, global row: %i, global column: %i, local row: %i, local column: %i \n", rank, size, M, N,m,n);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    // -----------------------------------------------------------------------
    // input: spRHS gives selected indices; output: spRHS holds selected entries of inv(A)
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\nSolve for selected entries of the inverse! \n---------------\n");CHKERRQ(ierr);
    // Create spRHS = spRHST^T. Two matrices share internal matrix data structure
    // Creates a new matrix object that behaves like A'. The transpose A' is NOT actually formed!
    // Rather the new matrix object performs the matrix-vector product by using the MatMultTranspose() on the original matrix
    ierr = MatCreateTranspose(spRHST,&spRHS);CHKERRQ(ierr);


    // Get user-specified set of entries in inverse of A
    ierr = MatMumpsGetInverse(F,spRHS);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"\Here\n");CHKERRQ(ierr);
    //ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
    ierr = MatMumpsGetInverseTranspose(F,spRHST);CHKERRQ(ierr);

    //ierr = MatMumpsGetInverseTranspose(F,spRHST);CHKERRQ(ierr);
    // Print the result for the inverse to terminal 
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nspRHS:\n");CHKERRQ(ierr);
        ierr = MatView(spRHS, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    // Write the inverse matrix to file
    //ierr = PetscOptionsGetString(NULL, NULL, "-fout" ,outputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr); 
    //ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, outputfile[0], FILE_MODE_WRITE, &fd);CHKERRQ(ierr); 
    // Necessary to store a dense matrix in parallel.
    //ierr = PetscViewerPushFormat(fd,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
    //ierr = MatView(X, fd);CHKERRQ(ierr); 

    // Free data structures 
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);    
    ierr = MatDestroy(&spRHST);CHKERRQ(ierr);
    //ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
    // Destroys a PetscViewer.
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

    // Handels options to be called at the conclusion of the program, and calls MPI_FInalize().
    ierr = PetscFinalize();
    return 0;
}

