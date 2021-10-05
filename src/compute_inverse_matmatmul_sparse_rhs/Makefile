# all target is usually the first in the makefile, since if you just write make in command line,
# without specifying the target, it will build the first target. ll target denotes that if you invoke it,
# make will build all what's needed to make a complete build. This is usually a dummy target,
# which doesn't create any files, but merely depends on the other files
all: compute_inverse_sparse_rhs

# Define source c file 
SOURCESC_MatMatMul_sp 	= compute_inverse_sparse_rhs.c
OBJECTFILE_MatMatMul_sp	= $(SOURCESC_MatMatMul_sp:.c=.o)

# Additional flags for PETSc
CFLAGS 			=
FFLAGS 			=
CPPFLAGS 		=
FPPFLAGS 		=

# Include petsc variables and compilation rules 
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compile the c-code, link and create an exectueable
compute_inverse_sparse_rhs: $(OBJECTFILE_MatMatMul_sp)
	-$(CLINKER) -o compute_inverse_sparse_rhs $(OBJECTFILE_MatMatMul_sp) $(PETSC_MAT_LIB)
	${RM} $(OBJECTFILE_MatMatMul_sp)

