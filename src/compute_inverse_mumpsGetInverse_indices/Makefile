# all target is usually the first in the makefile, since if you just write make in command line,
# without specifying the target, it will build the first target. ll target denotes that if you invoke it,
# make will build all what's needed to make a complete build. This is usually a dummy target,
# which doesn't create any files, but merely depends on the other files
all: inverse_MatMumpsGetInverse_indices

# Set PETSc_dir und arch 
PETSC_DIR       	= $PETSC_DIR
PETSC_ARCH      	= $PETSC_ARCH

# Define source c file 
SOURCESC_MatMumpsGetInv	= inverse_MumpsGetInverse_indices.c
OBJECTFILE_MatMumpsGetInv= $(SOURCESC_MatMumpsGetInv:.c=.o)

# Additional flags for PETSc
CFLAGS 			=
FFLAGS 			=
CPPFLAGS 		=
FPPFLAGS 		=

# Include petsc variables and compilation rules 
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compile the c-code, link and create an exectueable
inverse_MatMumpsGetInverse_indices: $(OBJECTFILE_MatMumpsGetInv)
	-$(CLINKER) -o inverse_MatMumpsGetInv_indices $(OBJECTFILE_MatMumpsGetInv) $(PETSC_MAT_LIB)
	${RM} $(OBJECTFILE_MatMumpsGetInv)

