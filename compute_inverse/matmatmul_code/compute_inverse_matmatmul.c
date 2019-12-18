static char help[] ="Compute a part of the inverse of a sparse matrix. This code requires that PETSc was configured with MUMPS since we are dealing with large matrices \
		     and therefore use a parallel LU factorization. We compute the inverse by solving the equation A*X=RHS. Where A is our Matrix, X is the inverse and RHS is the identity matrix.\
		     Note that the number of columns nrhs of X can be chosen smaller than the number of columns N in A. Therefore only a part of the inverse is computed in X. \n \
		     Input parameters include\n\
  			-fin <input_file> : file to load \n \
                	-fout <input_file> : file to load \n \
			-nrhs <numberofcolumns> : Number of columns to be compute \n \
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
    Mat			A,F,RHS,X;				// Abstract PETSc matrix object used to manage all linear operators in PETSc
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
    
    // Returns the numbers of rows and columns in a matrix.
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
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for Matrix A, rank:  %i, size: %i, rstart: %i, rend: %i \n", rank, size, rstart, rend);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    //Gets the integer value for a particular option in the database.
    // Set the number of colums on the right hand side
    nrhs = N;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, &flg2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Compute %i columns of the inverse\n ", nrhs);

    // Create dense matrix B and X; Using for A*X=RHS, where RHS is a diagonal matrix and X is empty
    ierr = MatCreate(PETSC_COMM_WORLD, &X);CHKERRQ(ierr);
    // Sets the local and global sizes, and checks to determine compatibility.
    ierr = MatSetSizes(X, PETSC_DECIDE, PETSC_DECIDE, M, nrhs);CHKERRQ(ierr);
    // Builds matrix object for a particular matrix type.
    ierr = MatSetType(X, MATDENSE);CHKERRQ(ierr);
    // Creates a matrix where the type is determined from the options database. Generates a parallel MPI matrix if the communicator has more than one processor.
    ierr = MatSetFromOptions(X);CHKERRQ(ierr);
    // Sets up the internal matrix data structures for the later use.
    ierr = MatSetUp(X);CHKERRQ(ierr);
    // The matrix entries are zero, so we do not set any values 
    // Begins assembling the matrix. 
    ierr = MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // Generate the dense matrix RHS which is a diagonal matrix
    ierr = MatCreate(PETSC_COMM_WORLD, &RHS);CHKERRQ(ierr);
    // Sets the local and global sizes, and checks to determine compatibility.
    ierr = MatSetSizes(RHS, PETSC_DECIDE, PETSC_DECIDE, M, nrhs);CHKERRQ(ierr);
    // Builds matrix object for a particular matrix type.
    ierr = MatSetType(RHS, MATDENSE);CHKERRQ(ierr);
    // Creates a matrix where the type is determined from the options database. Generates a parallel MPI matrix if the communicator has more than one processor.
    ierr = MatSetFromOptions(RHS);CHKERRQ(ierr);
    // Sets up the internal matrix data structures for the later use.
    ierr = MatSetUp(RHS);CHKERRQ(ierr);

    // Insert ones on the diagonal
    ierr = MatGetOwnershipRange(RHS,&rstart,&rend);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for Matrix RHS, rank:  %i, size: %i, rstart: %i, rend: %i \n", rank, size, rstart, rend);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    if (rstart < nrhs  && rend <= nrhs){
        for (i=rstart; i<rend; i++){
            v = 1.0;
            ierr = MatSetValues(RHS, 1, &i, 1, &i, &v, INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    if(rstart < nrhs && rend >= nrhs){
        for (i=rstart; i<nrhs; i++){
            v = 1.0;
            ierr = MatSetValues(RHS, 1, &i, 1, &i, &v, INSERT_VALUES);CHKERRQ(ierr);
        }
    }

    // Begins and ends assembling the matrix. 
    ierr = MatAssemblyBegin(RHS, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(RHS, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);    

    // Print matrix to terminal
    if (displ){
        ierr = MatView(RHS, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    

    // Factorize the Matrix using a parallel LU factorization in MUMPS
    // Cholesky is not possible since we are dealing with postivie semi-definite matrices --> Could be numerically unstable!
    PetscPrintf(PETSC_COMM_WORLD, "Use LU factorization! \n");
    // Returns a matrix suitable to calls to MatXXFactorSymbolic().
    ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &F);CHKERRQ(ierr);
    // Performs symbolic LU factorization of matrix.
    ierr = MatLUFactorSymbolic(F, A, NULL, NULL, NULL);CHKERRQ(ierr);
    // Performs numeric LU factorization of a matrix.
    ierr = MatLUFactorNumeric(F, A, NULL);CHKERRQ(ierr);

    // Compute the inverse by using MatMatSolve. Solves A*X= RHS
    ierr = MatMatSolve(F, RHS, X);

    // Print the result for the inverse to terminal 
    if (displ){
        ierr = MatView(X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    // Write the inverse matrix to file
    ierr = PetscOptionsGetString(NULL, NULL, "-fout" ,outputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr); 
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, outputfile[0], FILE_MODE_WRITE, &fd);CHKERRQ(ierr); 
    // Necessary to store a dense matrix in parallel.
    ierr = PetscViewerPushFormat(fd,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
    ierr = MatView(X, fd);CHKERRQ(ierr); 

    // Free data structures 
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);    
    ierr = MatDestroy(&RHS);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr); 
    // Destroys a PetscViewer.
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

    // Handels options to be called at the conclusion of the program, and calls MPI_FInalize().
    ierr = PetscFinalize();
    return 0;
}

