# all target is usually the first in the makefile, since if you just write make in command line,
# without specifying the target, it will build the first target. ll target denotes that if you invoke it,
# make will build all what's needed to make a complete build. This is usually a dummy target,
# which doesn't create any files, but merely depends on the other files
all: compute_inverse_mumps compute_inverse_matmatmul

# Set PETSc_dir und arch 
PETSC_DIR       	= /home/fr/fr_fr/fr_jg1080/Libaries/petsc_3.12
PETSC_ARCH      	= arch-linux-c-debug

# Define source c file 
SOURCESC		= save_example_matrix.c
OBJECTFILE		= $(SOURCESC:.c=.o)


# Additional flags for PETSc
CFLAGS 			=
FFLAGS 			=
CPPFLAGS 		=
FPPFLAGS 		=

# Include petsc variables and compilation rules 
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# Compile the c-code, link and create an exectueable
# Save matrix file 
save_file: $(OBJECTFILE)  
	-$(CLINKER) -o save_file $(OBJECTFILE) $(PETSC_MAT_LIB)
	${RM} $(OBJECTFILE_MUMPS)

load_file: load_file.o
	-$(CLINKER) -o load_file load_file.o $(PETSC_MAT_LIB)
	${RM} load_file.o

