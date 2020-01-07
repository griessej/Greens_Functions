# all target is usually the first in the makefile, since if you just write make in command line,
# without specifying the target, it will build the first target. ll target denotes that if you invoke it,
# make will build all what's needed to make a complete build. This is usually a dummy target,
# which doesn't create any files, but merely depends on the other files
all: compute_residual_of_inverse

# Set PETSc_dir und arch 
PETSC_DIR       	= /home/fr/fr_fr/fr_jg1080/Libaries/petsc_3.12.0_debugging_0
PETSC_ARCH      	= arch-linux-c-opt

# Define source c file 
SOURCESC_Residual	= compute_residual_of_inverse.c
OBJECTFILE_Residual	= $(SOURCESC_Residual:.c=.o)

# Additional flags for PETSc
CFLAGS 			=
FFLAGS 			=
CPPFLAGS 		=
FPPFLAGS 		=

# Include petsc variables and compilation rules 
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compile the c-code, link and create an exectueable
compute_residual_of_inverse: $(OBJECTFILE_Residual)
	-$(CLINKER) -o compute_residual_of_inverse $(OBJECTFILE_Residual) $(PETSC_MAT_LIB)
	${RM} $(OBJECTFILE_Residual)

