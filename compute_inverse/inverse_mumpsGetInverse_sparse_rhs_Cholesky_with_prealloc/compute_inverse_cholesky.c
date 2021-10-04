static char help[] ="\
This program computes the inverse of a sparse, symmetric, positive-definite matrix A. \
Optionally, it computes only the upper left (0, rhs_col) × (0, rhs_row) block. \
The inverse is determined using Cholesky factorization in MUMPS. \n\
Options: \n\
 -fin  <file> : Input file \n\
 -fout <file> : Output file \n\
 -rhs_col <numberofcolumns> : number of columns to compute \n\
 -rhs_row <numberofrows> : number of rows to compute \n\
 -displ <Bool>: print matrices to terminal? \n\
 \n\
Example usage: \n \
mpiexec -np 2 ./compute_inverse_cholesky \\\n\
    -fin  10x10_matrix.mat \\\n\
    -fout 5x5_block_of_inverse.mat \\\n\
    -rhs_row 5 -rhs_col 5 -displ \n\
\n\
Use option -log_view for profiling";

#include <stdio.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscviewer.h> 

int main(int argc, char **args){
    PetscErrorCode  ierr;                    
    PetscMPIInt     size,rank;               
#if defined(PETSC_HAVE_MUMPS)
    Mat             A,F,spRHS;               // PETSc matrices
    MatInfo         matinfo;
    PetscViewer     fd;                      
    PetscBool       flg;          
    PetscBool       displ=PETSC_FALSE;       // Display matrices if set to True otherwise False
    PetscInt        M,N,m,n,rstart,rend,i,j; 
    PetscInt        FM, FN;                  // Shape of factor matrix F
    PetscInt        spRHSM, spRHSN;          // Shape of right-hand side matrix spRHS
    PetscInt        rhs_col,rhs_row;         // Number of columns and rows on RHS (right-hand side)
    PetscInt        total_nz;                // Number of nonzero values to be pre-allocated in spRHS
    PetscBool       prealloc=PETSC_TRUE;     // Display matrices if set to True otherwise False
    PetscReal       symtol=PETSC_SQRT_MACHINE_EPSILON;
    MatType mtype;
    MatFactorInfo   factinfo;
    IS              isrow,iscol;            /* row and column permutations */
    char inputfile[1][PETSC_MAX_PATH_LEN];   // Input file name 
    char outputfile[1][PETSC_MAX_PATH_LEN];  // Outputfile file name 
#endif

    ierr = PetscInitialize(&argc, &args, (char*)0, help);if (ierr){return ierr;}
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

#if !defined(PETSC_HAVE_MUMPS)
    if (!=rank){ierr = PetscPrintf(PETSC_COMM_SELF, "ERROR: this code requires MUMPS.\n");CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
    }
#else
    ierr = PetscOptionsGetBool(NULL, NULL, "-displ", &displ, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-fin" ,inputfile[0], PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
    if (!flg) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "ERROR: input file must be specified using -fin");
    }
    ierr = PetscOptionsGetString(NULL, NULL, "-fout" ,outputfile[0], PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr); 
    if (!flg) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "ERROR: output file must be specified using -fout");
    }

    // Load matrix A from file 
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Loading matrix in file %s \n", inputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, inputfile[0], FILE_MODE_READ, &fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);  
    //ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);  
    ierr = MatLoad(A, fd);CHKERRQ(ierr);

    //ierr = PetscPrintf("The size of PetscInt is %d\n", sizeof(PetscInt));CHKERRQ(ierr);

    // Print matrix A if -displ is set. 
    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Input matrix A:\n");
        ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    ierr = MatGetType(A, &mtype);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Input matrix A is of type %s\n", mtype);CHKERRQ(ierr);
    // Check if matrix is quadratic
    ierr = MatGetSize(A, &M, &N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A, &m, &n);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Input matrix A has shape (%d, %d)\n", M, N);CHKERRQ(ierr);
    if (M != N){
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "ERROR: input matrix must be square!");
    }
    //ierr = MatIsSymmetric(A,symtol,&flg);CHKERRQ(ierr);
    //if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A must be symmetric for Cholesky factorization");
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Ownership ranges for input matrix A:\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "rank, size, rstart, rend, M, N, m, n\n");CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%i %i %i %i %i %i %i %i \n", rank, size, rstart, rend, M, N, m, n);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);CHKERRQ(ierr);

    rhs_col = N;
    rhs_row = M;
    ierr = PetscOptionsGetInt(NULL, NULL, "-rhs_col", &rhs_col, &flg);CHKERRQ(ierr);
    if (flg){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Will calculate first %d columns of the inverse of A\n", rhs_col);
    }
    else{
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Will calculate all %d columns of the inverse of A.\n", rhs_row);
    }
    ierr = PetscOptionsGetInt(NULL, NULL, "-rhs_row", &rhs_row, &flg);CHKERRQ(ierr);
    if (flg){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Will calculate first %d rows of the inverse of A.\n", rhs_row);
    }
    else{
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Will calculate all %d rows of the inverse of A.\n", rhs_row);
    }
    total_nz = rhs_col;
    MatGetInfo(A,MAT_LOCAL,&matinfo);
    if (!rank){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo of A on rank 0:\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.block_size       : %f\n", matinfo.block_size       );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_allocated     : %f\n", matinfo.nz_allocated     );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_used          : %f\n", matinfo.nz_used          );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_unneeded      : %f\n", matinfo.nz_unneeded      );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.assemblies       : %f\n", matinfo.assemblies       );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.mallocs          : %f\n", matinfo.mallocs          );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.fill_ratio_given : %f\n", matinfo.fill_ratio_given );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.fill_ratio_needed: %f\n", matinfo.fill_ratio_needed);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.factor_mallocs   : %f\n", matinfo.factor_mallocs   );CHKERRQ(ierr);
    }
        
    // spRHST is a sparse matrix that stores the transpose of the inverse.
    // Notice that MUMPS expects the right hand side to be in compressed
    // column format, but PETSc supports only compressed row format.
    // Furthermore, MUMPS requires nrhs=N. Therefore, we calculate the transpose
    // of the inverse in spRHST. It needs to be transposed afterwards.
    //
    // MUMPS requires the RHS to exist on MPI rank 0.
    ierr = MatCreate(PETSC_COMM_WORLD, &spRHS);CHKERRQ(ierr);
    if (!rank){
        //ierr = MatSetSizes(spRHS,M,N,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
        ierr = MatSetSizes(spRHS,M,N,M,N);CHKERRQ(ierr);
    }
    else{
        //ierr = MatSetSizes(spRHS,0,0,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
        ierr = MatSetSizes(spRHS,0,0,M,N);CHKERRQ(ierr);
    }
    ierr = MatSetType(spRHS,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(spRHS);CHKERRQ(ierr);
    if (prealloc){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Pre-allocating right hand side matrix spRHS, total_nz = %d\n",total_nz);
        if (!rank){
            ierr = MatMPIAIJSetPreallocation(spRHS, rhs_col,NULL, 0,NULL);CHKERRQ(ierr); 
        }
        else{
            ierr = MatMPIAIJSetPreallocation(spRHS, 0,NULL, 0,NULL);CHKERRQ(ierr); 
        }
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Calling MatSetUp on spRHS.\n");
    ierr = MatSetUp(spRHS);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Filling right hand side matrix spRHS.\n");
    if (!rank){
        // Value to fill into the matrix 
        PetscScalar v[rhs_col];
        // Global col indices
        PetscInt idxn[rhs_col];
        // Fill value array and create indices of columns and rows
        for(j = 0; j < rhs_col; j++){
            v[j] = 1.0; 
            idxn[j] = j;
        }  
        // Fill the upper left part of the matrix with 1.
        // Be careful, we fill the transposed matrix due to
        // compressed col format instead of compressed row format!
        for(i=0;i<rhs_row;i++){
            ierr = MatSetValues(spRHS,rhs_col,idxn,1,&i,v,INSERT_VALUES);CHKERRQ(ierr);       
        }
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Assembling right hand side matrix spRHS\n");CHKERRQ(ierr);
    ierr = MatAssemblyBegin(spRHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(spRHS, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatGetType(spRHS, &mtype);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "spRHS is of type %s\n", mtype);CHKERRQ(ierr);
    ierr = MatGetSize(spRHS, &spRHSM, &spRHSN);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "spRHS has shape (%d, %d)\n", spRHSM, spRHSN);CHKERRQ(ierr);
    MatGetInfo(spRHS,MAT_LOCAL,&matinfo);
    if (!rank){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo of spRHS on rank 0:\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.block_size       : %f\n", matinfo.block_size       );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_allocated     : %f\n", matinfo.nz_allocated     );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_used          : %f\n", matinfo.nz_used          );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_unneeded      : %f\n", matinfo.nz_unneeded      );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.assemblies       : %f\n", matinfo.assemblies       );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.mallocs          : %f\n", matinfo.mallocs          );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.fill_ratio_given : %f\n", matinfo.fill_ratio_given );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.fill_ratio_needed: %f\n", matinfo.fill_ratio_needed);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.factor_mallocs   : %f\n", matinfo.factor_mallocs   );CHKERRQ(ierr);
    }

    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Below is the right hand side matrix spRHS. Note that this matrix will have the same shape as the transpose of the requested block of the inverse.\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"The reason ist that MUMPS uses commpressed column format, instead of petsc commpressed row format, for matrix storage.\n");CHKERRQ(ierr);
        ierr = MatView(spRHS, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    //ierr = MatFactorInfoInitialize(&factinfo);CHKERRQ(ierr);
    //factinfo.fill=5.0;
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.diagonal_fill: %f\n", factinfo.diagonal_fill);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.usedt:         %f\n", factinfo.usedt);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.dt:            %f\n", factinfo.dt);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.dtcol:         %f\n", factinfo.dtcol);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.fill:          %f\n", factinfo.fill);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.levels:        %f\n", factinfo.levels);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.pivotinblocks: %f\n", factinfo.pivotinblocks);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.zeropivot:     %f\n", factinfo.zeropivot);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.shifttype:     %f\n", factinfo.shifttype);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "MatFactorInfo.shiftamount:   %f\n", factinfo.shiftamount);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Computing %i columns and %i rows of the inverse using Cholesky factorization in MUMPS.\n", 
        rhs_col, rhs_row);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "Getting ordering isrow/iscol\n");CHKERRQ(ierr);
    //ierr = MatGetOrdering(A,MATORDERINGNATURAL, &isrow, &iscol);CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD, "Reordering for nonzero diagonal\n");CHKERRQ(ierr);
    //ierr = MatReorderForNonzeroDiagonal(A,1.e-8,isrow,iscol);CHKERRQ(ierr);
    ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &F);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Performing symbolic Cholesky factorization with MatCholeskyFactorSymbolic\n");CHKERRQ(ierr);
    //ierr = MatCholeskyFactorSymbolic(F, A, iscol, &factinfo);CHKERRQ(ierr);
    //ierr = MatCholeskyFactorSymbolic(F, A, NULL, &factinfo);CHKERRQ(ierr);
    ierr = MatCholeskyFactorSymbolic(F, A, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Performing numeric Cholesky factorization with MatCholeskyFactorNumeric\n");CHKERRQ(ierr);
    //ierr = MatCholeskyFactorNumeric(F, A, &factinfo);CHKERRQ(ierr);
    ierr = MatCholeskyFactorNumeric(F, A, NULL);CHKERRQ(ierr);
    ierr = MatGetType(F, &mtype);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Factor matrix F is of type %s\n", mtype);CHKERRQ(ierr);
    ierr = MatGetSize(F, &FM, &FN);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Factor matrix F has shape (%d, %d)\n", FM, FN);CHKERRQ(ierr);
    MatGetInfo(F,MAT_LOCAL,&matinfo);
    if (!rank){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo of F on rank 0:\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.block_size       : %f\n", matinfo.block_size       );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_allocated     : %f\n", matinfo.nz_allocated     );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_used          : %f\n", matinfo.nz_used          );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.nz_unneeded      : %f\n", matinfo.nz_unneeded      );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.assemblies       : %f\n", matinfo.assemblies       );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.mallocs          : %f\n", matinfo.mallocs          );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.fill_ratio_given : %f\n", matinfo.fill_ratio_given );CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.fill_ratio_needed: %f\n", matinfo.fill_ratio_needed);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "MatInfo.factor_mallocs   : %f\n", matinfo.factor_mallocs   );CHKERRQ(ierr);
    }

    // Create the tranpose of the right hand side matrix spRHS. From the manual: "The
    // transpose A' is NOT actually formed! Rather the new matrix object performs the
    // matrix-vector product by using the MatMultTranspose() on the original matrix."
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Creating transpose right hand side spRHST from spRHS\n");CHKERRQ(ierr);
    Mat spRHST = NULL;
    ierr = MatCreateTranspose(spRHS,&spRHST);CHKERRQ(ierr);

    // Get user-specified set of entries in inverse of A
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Computing inverse of A from factor matrix F using MatMumpsGetInverse. Storing result in spRHST.\n");CHKERRQ(ierr);
    ierr = MatMumpsGetInverse(F,spRHST);CHKERRQ(ierr);

    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"spRHST");CHKERRQ(ierr);
        ierr = MatView(spRHST, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
    }

    // Compute the transpose of the matrix
    ierr = MatTranspose(spRHS, MAT_INPLACE_MATRIX, &spRHS);CHKERRQ(ierr);

    if (displ){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n---------------\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"First %D columns and %D rows of inv(A) with sparse RHS:\n", rhs_col, rhs_row);
        ierr = MatView(spRHS,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "---------------\n");CHKERRQ(ierr);
    }
    
    // Write the inverse matrix to file
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Save inverse matrix in: %s \n", outputfile[0]);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, outputfile[0], FILE_MODE_WRITE, &fd);CHKERRQ(ierr); 
    ierr = MatView(spRHS, fd);CHKERRQ(ierr); 

    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
    ierr = MatDestroy(&spRHST);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
    
#endif
}


