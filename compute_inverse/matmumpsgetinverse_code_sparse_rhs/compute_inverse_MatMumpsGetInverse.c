static char help[] ="Compute a part of the inverse of a sparse matrix. This code requires that PETSc was configured with MUMPS since we are dealing with large matrices \
		     and therefore use a parallel LU factorization. We compute the inverse by solving the equation A*X=RHS. Where A is our Matrix, X is the inverse and RHS is the identity matrix.\
		     Note that the number of columns nrhs of X can be chosen smaller than the number of columns N in A. Therefore only a part of the inverse is computed in X. \n \
		     In this code we use a sparse representation of the RHS matrix in MUMPS in csr format. Computation of selected entries in inv(A) is done using MatMumpsGetInverse. \n \
		     Input parameters: \n\
  			-fin <input_file> : file to load \n \
                	-fout <input_file> : file to load \n \
			-nrhs <numberofcolumns> : Number of columns to compute \n \
                        -displ <Bool>: Print matrices to terminal \n\
			-checkResidual <Bool>: Check the residual R = A*A^-1 -diag(1) \n \
		     Example usage: \n \
		         mpiexec -np 2 ./compute_inverse_sparse_rhs -fin ../../convert_to_binary_petsc_matrix/identity_matrix_prefactor3_ncols10 -fout test -nrhs 5 -displ -checkResidual";

#include <stdio.h>
#include <petscmat.h>
#include <petscviewer.h> 

int main(int argc, char **args){
    PetscErrorCode 	ierr; 					// Datatype used for return error code
    PetscMPIInt		size,rank; 				// Datatype used to represent 'int' parameters to MPI functions.
#if defined(PETSC_HAVE_MUMPS)
    Mat			A,F,spRHST;				// Abstract PETSc matrix object used to manage all linear operators in PETSc
    PetscViewer		fd; 					// Abstract PETSc object that helps view (in ASCII, binary, graphically etc) other PETSc objects
    PetscBool      	flg1,flg2;				// Logical variable. Actually an int in C.
    PetscBool		displ=PETSC_FALSE;			// Display matrices if set to True otherwise False
    PetscBool		checkResidual=PETSC_FALSE;		// Check the residual of the the computed inverse 
    PetscInt		M,N,m,n,rstart,rend,nrhs,i,j;		// PETSc type that represents an integer, used primarily to represent size of arrays and indexing into arrays.
    PetscReal      	norm,tol=PETSC_SQRT_MACHINE_EPSILON;    // PETSc type that represents a real number version of PetscScalar
    char		inputfile[1][PETSC_MAX_PATH_LEN]; 	// Input file name 
    char		outputfile[1][PETSC_MAX_PATH_LEN]; 	// Outputfile file name 
#endif

    // Initializes PETSc and MPI. Get size and rank of MPI.
    ierr = PetscInitialize(&argc, &args, (char*)0, help);if (ierr){return ierr;}
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

    //Check if PETSc was configured with MUMPS. If not print error message and exit 
#if !defined(PETSC_HAVE_MUMPS)
    if (!=rank){ierr = PetscPrintf(PETSC_COMM_SELF, "This code requires MUMPS, exit...\n");CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
    }
#else

    // Check if displ is set. If True the matrices are printed to the terminal
    ierr = PetscOptionsGetBool(NULL, NULL, "-displ", &displ, NULL);CHKERRQ(ierr);
    // Check if checkResidual is set. If True compute the R = A*A^-1 -diag(1)
    ierr = PetscOptionsGetBool(NULL, NULL, "-checkResidual", &checkResidual, NULL);CHKERRQ(ierr);

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
        
    // Create SpRHST for inv(A) with sparse RHS stored in the host.
    // PETSc does not support compressed column format which is required by MUMPS for sparse RHS matrix,
    // thus user must create spRHST=spRHS^T and call MatMatTransposeSolve()
    // User must create B^T in sparse compressed row format on the host processor and call MatMatTransposeSolve() to implement MUMPS' MatMatSolve().
    // MUMPS requires nrhs = N 
    ierr = MatCreate(PETSC_COMM_WORLD, &spRHST);CHKERRQ(ierr);
    if (!rank){
        ierr = MatSetSizes(spRHST,N,M,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    else{
        ierr = MatSetSizes(spRHST,0,0,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = MatSetType(spRHST,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(spRHST);CHKERRQ(ierr);
    ierr = MatSetUp(spRHST);CHKERRQ(ierr);
    if (!rank){
        // PETSc type that represents either a double precision real number,...
        PetscScalar v[nrhs];
        PetscInt idxn[nrhs];
        // Generate the column indices 
        for(j = 0; j < nrhs; j++){
            idxn[j] = j;
            v[j] = 1.0;
        }  
        for(i=0;i<nrhs;i++){
            ierr = MatSetValues(spRHST,1,&i,nrhs,idxn,v,INSERT_VALUES);CHKERRQ(ierr);     
        }
    }
    ierr = MatAssemblyBegin(spRHST,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(spRHST, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // Print matrix spRHST 
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix spRHST:\n", nrhs);
        ierr = MatView(spRHST, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }

    // Print information
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nCompute %i columns of the inverse using LU-factorization in MUMPS!\n", nrhs);

    // Factorize the Matrix using a parallel LU factorization in MUMPS
    ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, &F);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(F, A, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatLUFactorNumeric(F, A, NULL);CHKERRQ(ierr);

    // Create spRHS 
    Mat spRHS = NULL;
    
    // Create spRHS = spRHS^T. Two matrices that share internal matrix data structure. 
    // Creates a new matrix object that behaves like A'.
    ierr = MatCreateTranspose(spRHST,&spRHS);CHKERRQ(ierr);

    // Get user-specified set of entries in inverse of A
    ierr = MatMumpsGetInverse(F,spRHS);CHKERRQ(ierr);

    // Compute the transpose of the matrix 
    ierr = MatTranspose(spRHST, MAT_INPLACE_MATRIX, &spRHST);CHKERRQ(ierr);

    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"First %D columns of inv(A) with sparse RHS:\n", nrhs);
        ierr = MatView(spRHST,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);
    }

    if (checkResidual){
        Mat AinvA,identityMatrix,invA;
        PetscScalar p;
        // Set up the identity matrix
        ierr = MatCreate(PETSC_COMM_WORLD,&identityMatrix);CHKERRQ(ierr);
        ierr = MatSetType(identityMatrix, MATAIJ);CHKERRQ(ierr);
        ierr = MatSetSizes(identityMatrix,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
        ierr = MatSetFromOptions(identityMatrix);CHKERRQ(ierr);
        ierr = MatSetUp(identityMatrix);CHKERRQ(ierr);

        ierr = MatGetOwnershipRange(identityMatrix,&rstart,&rend);CHKERRQ(ierr);
        if (rstart < nrhs  && rend <= nrhs){
            for (i=rstart; i<rend; i++){
                p = 1.0;
                ierr = MatSetValues(identityMatrix, 1, &i, 1, &i, &p, INSERT_VALUES);CHKERRQ(ierr);
            }
        }
        if(rstart < nrhs && rend >= nrhs){
            for (i=rstart; i<nrhs; i++){
                p = 1.0;
                ierr = MatSetValues(identityMatrix, 1, &i, 1, &i, &p, INSERT_VALUES);CHKERRQ(ierr);
            }
         }
         // Assemble matrix 
        ierr = MatAssemblyBegin(identityMatrix, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(identityMatrix, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  

        /*
        ierr = MatCreate(PETSC_COMM_WORLD,&invA);CHKERRQ(ierr);
        ierr = MatSetType(invA, MATAIJ);CHKERRQ(ierr);
        ierr = MatSetSizes(invA,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
        ierr = MatSetFromOptions(invA);CHKERRQ(ierr);
        ierr = MatSetUp(invA);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(invA, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(invA, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

        // copy values
        ierr = MatCopy(spRHST,invA,SAME_NONZERO_PATTERN);
 
        //ierr = MatConvert(spRHST,MATSAME,MAT_INITIAL_MATRIX,&invA);CHKERRQ(ierr); 
*/
        // Check the residual: R = A*A^-1 - diag(1)
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\nHere!\n");
        ierr = MatMatMult(A,spRHST,MAT_INITIAL_MATRIX,2.0,&AinvA);CHKERRQ(ierr);
        ierr = MatAXPY(AinvA,-1.0,identityMatrix,SAME_NONZERO_PATTERN);CHKERRQ(ierr); 
        ierr = MatNorm(AinvA,NORM_INFINITY,&norm);CHKERRQ(ierr);
        if (norm > tol){
            ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of the residual bigger than tolerance(=machine epsilon): %g\n",norm);CHKERRQ(ierr);
        }
        else{
            ierr = PetscPrintf(PETSC_COMM_SELF,"Norm of the residual smaller than tolerance(=machine epsilon)");CHKERRQ(ierr);
        }

    }
    
    // Write the inverse matrix to file
    ierr = PetscOptionsGetString(NULL, NULL, "-fout" ,outputfile[0], PETSC_MAX_PATH_LEN, &flg1);CHKERRQ(ierr); 
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nSave inverse matrix in: %s \n", outputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, outputfile[0], FILE_MODE_WRITE, &fd);CHKERRQ(ierr); 
    ierr = MatView(spRHST, fd);CHKERRQ(ierr); 

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


