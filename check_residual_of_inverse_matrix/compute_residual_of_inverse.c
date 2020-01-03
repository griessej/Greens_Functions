static char help[] ="Compute the residual R= A*A^-1 - diag(1).\n \
		     Input parameters: \n\
  			-fin_A <input_file> : Matrix A \n \
                	-fin_Ainv <input_file> : Matrix A^-1. This matrix is not necessarly the full inverse of A. \n \
			-nrhs <numberofcolumns> : Number of columns to compute \n \
                        -displ <Bool>: Print matrices to terminal \n\
		     Example usage: \n \
		          mpiexec -np 2 compute_residual_of_inverse -fin_A ../convert_to_binary_petsc_matrix/main_off_diagonal_matrix_offValue2_ncols10_full/main_off_diagonal_matrix_offValue2_ncols10 -fin_Ainv ../compute_inverse/code_inverse_matmumpsgetinverse_sparse_rhs/test -nrhs 5 -displ";

#include <stdio.h>
#include <petscmat.h>
#include <petscviewer.h> 

int main(int argc, char **args){
    PetscErrorCode 	ierr; 						// Datatype used for return error code
    PetscMPIInt		size,rank; 					// Datatype used to represent 'int' parameters to MPI functions.
    Mat			A,Ainv,identityMatrix,AAinv;			// Abstract PETSc matrix object used to manage all linear operators in PETSc
    PetscViewer		fd; 						// Abstract PETSc object that helps view (in ASCII, binary, graphically etc) other PETSc objects
    PetscBool      	flg1,flg2,flg3;					// Logical variable. Actually an int in C.
    PetscBool		displ=PETSC_FALSE;				// Display matrices if set to True otherwise False
    PetscInt		nrhs,i;						// PETSc type that represents an integer, used primarily to represent size of arrays and indexing into arrays.
    PetscInt		MA,NA,mA,nA,rstartA,rendA;			// Variables for matrix A 
    PetscInt		MAinv,NAinv,mAinv,nAinv,rstartAinv,rendAinv;	// Variables for matrix A^-1
    PetscInt		MId,NId,mId,nId,rstartId,rendId;		// Variables for identity matrix 
    PetscScalar    	v;						// PETSc type that represents either a double precision real number, a double precision complex number, ...
    PetscReal      	norm,tol=PETSC_SQRT_MACHINE_EPSILON;    	// PETSc type that represents a real number version of PetscScalar. 
    char		A_inputfile[1][PETSC_MAX_PATH_LEN]; 		// Input filename of the matrix A  
    char		Ainv_inputfile[1][PETSC_MAX_PATH_LEN]; 		// Input filename of the matrix A^-1

    // Initializes PETSc and MPI. Get size and rank of MPI.
    ierr = PetscInitialize(&argc, &args, (char*)0, help);if (ierr){return ierr;}
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

    // Check if displ is set. If True the matrices are printed to screen.
    ierr = PetscOptionsGetBool(NULL, NULL, "-displ", &displ, NULL);CHKERRQ(ierr);

    // Load matrix A from file 
    ierr = PetscOptionsGetString(NULL, NULL, "-fin_A" , A_inputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Load matrix A in: %s \n", A_inputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, A_inputfile[0], FILE_MODE_READ, &fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);  
    ierr = MatLoad(A, fd);CHKERRQ(ierr);
    // Print matrix A 
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix A from file:\n");
        ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }
    // Check if matrix is quadratic
    ierr = MatGetSize(A, &MA, &NA);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &mA, &nA);CHKERRQ(ierr);
    if (MA != NA){
        //Macro that is called when an error has been detected.
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Detected a rectangular matrix: (%d, %d)", MA, NA);
    }
    // Print rank, size, ownership, local and global matrix parameters to screen
    ierr = MatGetOwnershipRange(A,&rstartA,&rendA);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for Matrix A, rank:  %i, size: %i, rstart: %i, rend: %i, M: %i, N: %i, m: %i, n: %i \n", rank, size, rstartA, rendA, MA, NA, mA, nA);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);


    // Load matrix A^-1 from file 
    ierr = PetscOptionsGetString(NULL, NULL, "-fin_Ainv", Ainv_inputfile[0], PETSC_MAX_PATH_LEN, &flg2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Load matrix A^-1 in: %s \n", Ainv_inputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, Ainv_inputfile[0], FILE_MODE_READ, &fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &Ainv);CHKERRQ(ierr);
    ierr = MatSetType(Ainv, MATAIJ);CHKERRQ(ierr);
    ierr = MatLoad(Ainv, fd);CHKERRQ(ierr);
    // Print matrix A^-1
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix A^-1 from file:\n");
        ierr = MatView(Ainv, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }
    // Check if matrix is quadratic
    ierr = MatGetSize(Ainv, &MAinv, &NAinv);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Ainv, &mAinv, &nAinv);CHKERRQ(ierr);
    if (MAinv != NAinv){
        //Macro that is called when an error has been detected.
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Detected a rectangular matrix: (%d, %d)", MAinv, NAinv);
    }
    // Print rank, size, ownership, local and global matrix parameters to screen
    ierr = MatGetOwnershipRange(Ainv,&rstartAinv,&rendAinv);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for Matrix Ainv, rank:  %i, size: %i, rstart: %i, rend: %i, M: %i, N: %i, m: %i, n: %i \n", rank, size, rstartAinv, rendAinv, MAinv, NAinv, mAinv, nAinv);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);

    // Check that the matrices A and Ainv have same shape 
    if (MA != MAinv){
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "A and Ainv have different number of global rows: (%d, %d)", MA, MAinv);
    }
    if (NA != NAinv){
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "A and Ainv have different number of global columns: (%d, %d)", NA, NAinv);
    }   

    // We are, in general, intersted only in a part of the inverse especially in the columns/rows for (0 to nrhs)/(0 o nrhs)
    nrhs = NA;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, &flg3);CHKERRQ(ierr);

    // Create an identity matrix
    ierr = MatCreate(PETSC_COMM_WORLD, &identityMatrix);CHKERRQ(ierr);
    ierr = MatSetSizes(identityMatrix, PETSC_DECIDE, PETSC_DECIDE, MA, NA);CHKERRQ(ierr);
    ierr = MatSetType(identityMatrix, MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(identityMatrix);CHKERRQ(ierr);
    ierr = MatSetUp(identityMatrix);CHKERRQ(ierr);

    // Insert ones on the diagonal
    ierr = MatGetSize(identityMatrix, &MId, &NId);CHKERRQ(ierr);
    ierr = MatGetLocalSize(identityMatrix, &mId, &nId);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(identityMatrix,&rstartId,&rendId);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);
    if (rstartId < nrhs  && rendId <= nrhs){
        for (i=rstartId; i<rendId; i++){
            v = 1.0;
            ierr = MatSetValues(identityMatrix, 1, &i, 1, &i, &v, INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    if(rstartId < nrhs && rendId >= nrhs){
        for (i=rstartId; i<nrhs; i++){
            v = 1.0;
            ierr = MatSetValues(identityMatrix, 1, &i, 1, &i, &v, INSERT_VALUES);CHKERRQ(ierr);
        }
    }

    // Begins and ends assembling the matrix. 
    ierr = MatAssemblyBegin(identityMatrix, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(identityMatrix, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Ownership ranges for identity Matrix, rank:  %i, size: %i, rstart: %i, rend: %i, M: %i, N: %i, m: %i, n: %i  \n", rank, size, rstartId, rendId, MId, NId, mId, nId);
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix diag(1) from file:\n");
        ierr = MatView(identityMatrix, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }


    // Check the residual: R = A*A^-1 - diag(1) 
    // Compute AAinv = A*A^-1
    ierr = MatMatMult(A, Ainv, MAT_INITIAL_MATRIX, 2.0, &AAinv);CHKERRQ(ierr); 

    // Print AA^-1
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix AA^-1:\n");
        ierr = MatView(AAinv, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }

    // Compute AAinv = -1*diag(1)  + AAinv
    ierr = MatAXPY(AAinv,-1.0,identityMatrix,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    
    // Print -1diag(1)+AA^-1
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix AA^-1 -1diag(1):\n");
        ierr = MatView(AAinv, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }
 
    ierr = MatNorm(AAinv, NORM_INFINITY, &norm);CHKERRQ(ierr); 
    if (norm >= tol){
        ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of residual is larger than tolerance (norm, tolerance): %g , %g \n", norm, tol);CHKERRQ(ierr);
    }
    else{
        ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of residual is smaller than tolerance (norm, tolerance): %g , %g \n",norm, tol );CHKERRQ(ierr);
    }

    // Free data structures 
    ierr = MatDestroy(&A);CHKERRQ(ierr); 
    ierr = MatDestroy(&Ainv);CHKERRQ(ierr); 
    ierr = MatDestroy(&AAinv);CHKERRQ(ierr); 
    ierr = MatDestroy(&identityMatrix);CHKERRQ(ierr); 
    // Finalize
    ierr = PetscFinalize();
    return ierr;
}

