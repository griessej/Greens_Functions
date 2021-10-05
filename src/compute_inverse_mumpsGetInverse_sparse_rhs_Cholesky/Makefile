# all target is usually the first in the makefile, since if you just write make in command line,
# without specifying the target, it will build the first target. ll target denotes that if you invoke it,
# make will build all what's needed to make a complete build. This is usually a dummy target,
# which doesn't create any files, but merely depends on the other files
all: compute_inverse_cholesky

# Set PETSc_dir und arch 
PETSC_DIR       	= $PETSC_DIR
PETSC_ARCH      	= $PETSC_ARCH

# Define source c file 
SOURCESC_cholesky	= compute_inverse_cholesky.c
OBJECTFILE_cholesky= $(SOURCESC_cholesky:.c=.o)

# Additional flags for PETSc
CFLAGS 			= -O3 -march=native -mtune=native
FFLAGS 			= -O3 -march=native -mtune=native
CPPFLAGS 		= -O3 -march=native -mtune=native
FPPFLAGS 		=

# Include petsc variables and compilation rules 
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compile the c-code, link and create an exectueable
compute_inverse_cholesky: $(OBJECTFILE_cholesky)
	-$(CLINKER) -o compute_inverse_cholesky $(OBJECTFILE_cholesky) $(PETSC_MAT_LIB)
	${RM} $(OBJECTFILE_cholesky)

