static char help[] ="Compute a part of the inverse of a sparse matrix. This code requires that PETSc was configured with MUMPS. \n \
		     MUMPS is only used in this code to do the LU factorization in parallel. \n \
		     Input parameters include\n\
  			-f <input_file> : file to load \n \
			-n <numberofcolumns> : Number of columns to be compute \n \
                        -displ ";

#include <stdio.h>
#include <petscmat.h>
#include <petscviewer.h> 

int main(int argc, char **args){
    PetscErrorCode 	ierr; 				// Datatype used for return error code
    PetscMPIInt		size,rank; 			// Datatype used to represent 'int' parameters to MPI functions.
#if defined(PETSC_HAVE_MUMPS)
    Mat			A,F,RHS,X;			// Abstract PETSc matrix object used to manage all linear operators in PETSc
    PetscViewer		fd; 				// Abstract PETSc object that helps view (in ASCII, binary, graphically etc) other PETSc objects
    PetscBool      	flg1,flg2,flg_symmetric;	// Logical variable. Actually an int in C.
    PetscBool		flg_mumps_lu=PETSC_FALSE;	// LU Factorization
    PetscBool		flg_mumps_ch=PETSC_FALSE;	// Cholesky Factorization
    PetscBool		displ=PETSC_FALSE;		// Display matrices 
    PetscInt		M,N,m,n,nrhs,rstart,rend,i;	// PETSc type that represents an integer, used primarily to represent size of arrays and indexing into arrays.
    PetscScalar         v;                              // PETSc type that represents either a double precision real number,...
    char		file[1][PETSC_MAX_PATH_LEN]; 	// Input file name 
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

    //Check if PETSc was configured with MUMPS. IF not exit
#if !defined(PETSC_HAVE_MUMPS)
    if (!=rank){
        ierr = PetscPrintf(PETSC_COMM_SELF, "This code requires MUMPS, exit...\n");
        CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
    }
#endif

    // Check if displ is set to True
    ierr = PetscOptionsGetBool(NULL, NULL, "-displ", &displ, NULL);CHKERRQ(ierr);

    // Load the matrix
    // Gets the string value for a particular option in the database.
    ierr = PetscOptionsGetString(NULL, NULL, "-f" ,file[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr);
    // Prints to standard out, only from the first processor in the communicator.
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Load matrix in: %s \n", file[0]);CHKERRQ(ierr);
    // Opens a file for binary input/output. 
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[0], FILE_MODE_READ, &fd);CHKERRQ(ierr);
    // Creates a matrix where the type is determined from either a call to MatSetType(), the options database or by reading a file.
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    // Builds matrix object for a particular matrix type. This matrix type is identical to MATSEQAIJ when constructed with a single process communicator, and MATMPIAIJ otherwise.
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);  
    // Loads a matrix that has been stored in binary/HDF5 format with MatView().
    ierr = MatLoad(A, fd);CHKERRQ(ierr);
    // Destroys a PetscViewer.
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

    // Returns the numbers of rows and columns in a matrix.
    ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
    // Returns the number of rows and columns in a matrix stored locally. m = number of local rows, n = number of local columns.
    ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
    //Gets the integer value for a particular option in the database.
    nrhs = N;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nrhs", &n, &flg2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Compute %i columns of the inverse\n ", n);
    // Check if matrix is quadratic
    if (M != N){
        //Macro that is called when an error has been detected.
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Detected a rectangular matrix: (%d, %d)", M, N);
    }

    // Create dense matrix B and X; Using for A*X=RHS, where RHS is a diagonal matrix and X is empty
    ierr = MatCreate(PETSC_COMM_WORLD, &X);CHKERRQ(ierr);
    // Sets the local and global sizes, and checks to determine compatibility.
    ierr = MatSetSizes(X, m, PETSC_DECIDE, PETSC_DECIDE, nrhs);CHKERRQ(ierr);
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
    ierr = MatSetSizes(RHS, m, PETSC_DECIDE, PETSC_DECIDE, nrhs);CHKERRQ(ierr);
    // Builds matrix object for a particular matrix type.
    ierr = MatSetType(RHS, MATDENSE);CHKERRQ(ierr);
    // Creates a matrix where the type is determined from the options database. Generates a parallel MPI matrix if the communicator has more than one processor.
    ierr = MatSetFromOptions(RHS);CHKERRQ(ierr);
    // Sets up the internal matrix data structures for the later use.
    ierr = MatSetUp(RHS);CHKERRQ(ierr);

    // Returns the range of matrix rows owned by this processor, assuming that the matrix is laid out with the first n1 rows on the first processor, the next n2 rows on the second, etc.
    ierr = MatGetOwnershipRange(RHS,&rstart,&rend);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank:  %i, size: %i, rstart: %i, rend: %i \n", rank, size, rstart, rend);
    // Inser ones on the diagonal
    for (i=rstart; i<rend; i++){
        v = 1.0;
        ierr = MatSetValues(RHS, 1, &i, 1, &i, &v, INSERT_VALUES);CHKERRQ(ierr);
    }

    // Begins assembling the matrix. 
    ierr = MatAssemblyBegin(RHS, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(RHS, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);    

    //Factorize the matrix using either LU or Cholesky Factorization
    ierr = PetscOptionsGetBool(NULL, NULL, "-use_mumps_lu", &flg_mumps_lu, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL, NULL, "-use_mumps_ch", &flg_mumps_ch, NULL);CHKERRQ(ierr);
    
    if (flg_mumps_lu == PETSC_TRUE){
        printf("Use LU factorization.");
        // Returns a matrix suitable to calls to MatXXFactorSymbolic().
        ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &F);CHKERRQ(ierr);
        // Performs symbolic LU factorization of matrix.
        ierr = MatLUFactorSymbolic(F, A, NULL, NULL, NULL);CHKERRQ(ierr);
        // Performs numeric LU factorization of a matrix.
        ierr = MatLUFactorNumeric(F, A, NULL);CHKERRQ(ierr);
    }
    if (flg_mumps_ch == PETSC_TRUE){
        printf("Use Cholesky factorization. Keep in mind only numerical stable for symmetric, positiv definite matrices!");
        // Test whether a matrix is symmetric
        ierr = MatIsSymmetric(A, 0.0, &flg_symmetric);CHKERRQ(ierr);
        if (!flg_symmetric){
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "A is not symmetric!");
        }
        // Returns a matrix suitable to calls to MatXXFactorSymbolic().
        ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &F);CHKERRQ(ierr);
        // Performs symbolic Cholesky factorization of matrix.
        ierr = MatCholeskyFactorSymbolic(F, A, NULL, NULL);CHKERRQ(ierr);
        // Performs numeric Cholesky factorization of a matrix.
        ierr = MatCholeskyFactorNumeric(F, A, NULL);CHKERRQ(ierr);
    }
    
    // Compute the inverse by using matmatmul
    
    // Free data structures 
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);    
    ierr = MatDestroy(&RHS);CHKERRQ(ierr);
    ierr = MatDestroy(&X);CHKERRQ(ierr); 

    // Handels options to be called at the conclusion of the program, and calls MPI_FInalize().
    ierr = PetscFinalize();
    return 0;
}

