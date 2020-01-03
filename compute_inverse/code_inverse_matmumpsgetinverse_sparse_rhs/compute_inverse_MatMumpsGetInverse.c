static char help[] ="Compute a part of the inverse of a sparse matrix. This code requires that PETSc was configured with MUMPS since we are dealing with large matrices \
		     and therefore use a parallel LU factorization. Furthermore computation of selected entries in inv(A) is done using MatMumpsGetInverse.\
		     Note that the number of columns nrhs of the inverse can be chosen smaller than the number of columns N in A. Therefore only a part of the inverse is computed.\n \
 		     We compute only a the part of the matrix in the columns/rows for (0,nrhs)/(0,nrhs).  \n\
		     Input parameters: \n\
  			-fin <input_file> : file to load \n \
                	-fout <input_file> : file to load \n \
			-nrhs <numberofcolumns> : Number of columns to compute \n \
                        -displ <Bool>: Print matrices to terminal \n\
		     Example usage: \n \
		          mpiexec -np 2 ./compute_inverse_MatMumpsGetInv -fin ../../convert_to_binary_petsc_matrix/main_off_diagonal_matrix_offValue2_ncols10_full/main_off_diagonal_matrix_offValue2_ncols10 -fout inverse_main_off_diagonal_matrix_offValue2_ncols10 -nrhs 5 -displ";

#include <stdio.h>
#include <petscmat.h>
#include <petscviewer.h> 

int main(int argc, char **args){
    PetscErrorCode 	ierr; 					// Datatype used for return error code
    PetscMPIInt		size,rank; 				// Datatype used to represent 'int' parameters to MPI functions.
#if defined(PETSC_HAVE_MUMPS)
    Mat			A,F,spRHS;				// Abstract PETSc matrix object used to manage all linear operators in PETSc
    PetscViewer		fd; 					// Abstract PETSc object that helps view (in ASCII, binary, graphically etc) other PETSc objects
    PetscBool      	flg1,flg2;				// Logical variable. Actually an int in C.
    PetscBool		displ=PETSC_FALSE;			// Display matrices if set to True otherwise False
    PetscInt		M,N,m,n,rstart,rend,nrhs,i,j;		// PETSc type that represents an integer, used primarily to represent size of arrays and indexing into arrays.
    char		inputfile[1][PETSC_MAX_PATH_LEN]; 	// Input file name 
    char		outputfile[1][PETSC_MAX_PATH_LEN]; 	// Outputfile file name 
#endif

    // Initializes PETSc and MPI. Get size and rank of MPI processes.
    ierr = PetscInitialize(&argc, &args, (char*)0, help);if (ierr){return ierr;}
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

    //Check if PETSc was configured with MUMPS. If not print error message and exit! 
#if !defined(PETSC_HAVE_MUMPS)
    if (!=rank){ierr = PetscPrintf(PETSC_COMM_SELF, "This code requires MUMPS, exit...\n");CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
    }
#else

    // Check if displ is set. If True the matrices are printed to screen.
    ierr = PetscOptionsGetBool(NULL, NULL, "-displ", &displ, NULL);CHKERRQ(ierr);

    // Load matrix A from file 
    ierr = PetscOptionsGetString(NULL, NULL, "-fin" ,inputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Load matrix in: %s \n", inputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, inputfile[0], FILE_MODE_READ, &fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);  
    ierr = MatLoad(A, fd);CHKERRQ(ierr);
    // Print matrix A if -displ is set. 
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
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for Matrix A, rank:  %i, size: %i, rstart: %i, rend: %i, M: %i, N: %i, m: %i, n: %i \n", rank, size, rstart, rend, M, N, m, n);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);

    // We are intersted only in a part of the inverse especially in the columns/rows for (0,nrhs)/(0,nrhs)
    nrhs = N;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, &flg2);CHKERRQ(ierr);
        
    // Create SpRHST for inv(A) with sparse RHS stored in the host.
    // PETSc does not support compressed column format which is required by MUMPS for sparse RHS matrix and furthermore MUMPS requires nrhs=N
    // Keep in mind that since MUMPS uses commpressed column format, instead of petsc commpressed row format, the inverse is transposed and needs to be transposed after computation!!
    // MUMPS requirs RHS be centralized on the host=rank0!!!! Matrix with global number of rows and columns on proc 0.
    ierr = MatCreate(PETSC_COMM_WORLD, &spRHS);CHKERRQ(ierr);
    if (!rank){
        ierr = MatSetSizes(spRHS,M,N,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    else{
        ierr = MatSetSizes(spRHS,0,0,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = MatSetType(spRHS,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(spRHS);CHKERRQ(ierr);
    ierr = MatSetUp(spRHS);CHKERRQ(ierr);
    if (!rank){
        PetscScalar v[nrhs];
        PetscInt idxn[nrhs];
        // Generate the column indices 
        for(j = 0; j < nrhs; j++){
            idxn[j] = j;
            v[j] = 1.0;
        }  
        // Fill the upper left part of the matrix with 1. Indicates that these entries of the inverse are computed.
        for(i=0;i<nrhs;i++){
            ierr = MatSetValues(spRHS,1,&i,nrhs,idxn,v,INSERT_VALUES);CHKERRQ(ierr);     
        }
    }
    ierr = MatAssemblyBegin(spRHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(spRHS, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // Print matrix spRHST 
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix spRHS:\n", nrhs);
        ierr = MatView(spRHS, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }

    // Print information
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCompute %i columns and rows of the inverse using LU-factorization in MUMPS!\n", nrhs);

    // Factorize the Matrix using a parallel LU factorization in MUMPS
    ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &F);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(F, A, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(F, A, NULL);CHKERRQ(ierr);

    // Create spRHST,  
    // The transpose A' is NOT actually formed! Rather the new matrix object performs the matrix-vector product by using the MatMultTranspose() on the original matrix.
    Mat spRHST = NULL;
    
    // Create spRHST = spRHS. Two matrices that share internal matrix data structure. 
    // Creates a new matrix object that behaves like A'.
    ierr = MatCreateTranspose(spRHS,&spRHST);CHKERRQ(ierr);

    // Get user-specified set of entries in inverse of A
    ierr = MatMumpsGetInverse(F,spRHST);CHKERRQ(ierr);

    // Compute the transpose of the matrix
    ierr = MatTranspose(spRHS, MAT_INPLACE_MATRIX, &spRHS);CHKERRQ(ierr);

    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"First %D columns of inv(A) with sparse RHS:\n", nrhs);
        ierr = MatView(spRHS,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);
    }
    
    // Write the inverse matrix to file
    ierr = PetscOptionsGetString(NULL, NULL, "-fout" ,outputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr); 
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nSave inverse matrix in: %s \n", outputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, outputfile[0], FILE_MODE_WRITE, &fd);CHKERRQ(ierr); 
    ierr = MatView(spRHS, fd);CHKERRQ(ierr); 

    // Free data structures
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
    ierr = MatDestroy(&spRHST);CHKERRQ(ierr);
    // Destroy the Viewer
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    // PETSc finalize
    ierr = PetscFinalize();
    return ierr;
    
#endif
}


