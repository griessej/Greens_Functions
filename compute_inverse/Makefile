# all target is usually the first in the makefile, since if you just write make in command line,
# without specifying the target, it will build the first target. ll target denotes that if you invoke it,
# make will build all what's needed to make a complete build. This is usually a dummy target,
# which doesn't create any files, but merely depends on the other files
all: compute_inverse_mumps compute_inverse_matmatmul

# Set PETSc_dir und arch 
PETSC_DIR       	= /home/fr/fr_fr/fr_jg1080/Libaries/petsc_3.12
PETSC_ARCH      	= arch-linux-c-debug

# Define source c file 
SOURCESC_MUMPS		= compute_inverse_mumps.c
SOURCESC_MatMatMul 	= compute_inverse_matmatmul.c
OBJECTFILE_MUMPS	= $(SOURCESC_MUMPS:.c=.o)
OBJECTFILE_MatMatMul	= $(SOURCESC_MatMatMul:.c=.o)

# Additional flags for PETSc
CFLAGS 			=
FFLAGS 			=
CPPFLAGS 		=
FPPFLAGS 		=

# Include petsc variables and compilation rules 
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compile the c-code, link and create an exectueable
# MUMPS usage of MumpsGetInverse
compute_inverse_mumps: $(OBJECTFILE_MUMPS)  
	-$(CLINKER) -o compute_inverse_mumps $(OBJECTFILE_MUMPS) $(PETSC_MAT_LIB)
	${RM} $(OBJECTFILE_MUMPS)

# MatMatSolve usage
compute_inverse_matmatmul: $(OBJECTFILE_MatMatMul)
	-$(CLINKER) -o compute_inverse_matmatmul $(OBJECTFILE_MatMatMul) $(PETSC_MAT_LIB)
	${RM} $(OBJECTFILE_MatMatMul)
