static char help[] = "Generate a 4x4 matrix and save it to a binary format.\n\
Example: mpiexec -n <np> ./ex214 -displ \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
    PetscErrorCode 	ierr;
    PetscMPIInt    	size,rank;
    PetscInt		m,n,M,N,Istart,Iend,Ii,J,j,i;
    Mat			A;
    PetscViewer		view;
    int            	fd;
    PetscScalar    	v;
    PetscBool      	displ=PETSC_FALSE;
    
    // Initialize PETSc
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;	
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

    ierr = PetscOptionsGetBool(NULL,NULL,"-displ",&displ,NULL);CHKERRQ(ierr);

    // Create Matrix 
    m = 4;
    n = 4;
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A, &Istart, &Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
        v = -1.0; i = Ii/n; j = Ii - i*n;
        if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

    if (displ){
        
        ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank: %i size: %i Istart: %i Iend: %i \n", rank, size, Istart, Iend);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
    ierr = PetscPrintf(PETSC_COMM_SELF,"M: %d, N: %d, m: %d, n: %d \n", M,N,m,n);CHKERRQ(ierr);

    //
    PetscViewerCreate(PETSC_COMM_WORLD, &view);
    PetscViewerSetType(view, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(view, FILE_MODE_WRITE);
    PetscViewerFileSetName(view, "sfm.mat");
    MatView(A, view);
    PetscViewerDestroy(&view);
    
    PetscFinalize();

}
