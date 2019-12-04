ALL: prog

PETSC_DIR 	= /Users/griesserj/Libaries/petsc
CFLAGS		= -I${PETSC_DIR}/include
FFLAGS		= -I${PETSC_DIR}/include/petsc/finclude
SOURCESC	= compute_inverse.c
SOURCESF 	=
OBJ			= ${SOURCESC: .c=.o}
CLEANFILE	= ${OBJ} compute_inverse.c

include $(PETSC_DIR)/lib/petsc/conf/variables
include $(PETSC_DIR)/lib/petsc/conf/rules

prog: ${OBJ} chkopts
	-${CLINKER} -o out.o ${OBJ} ${PETSC_SYS_LIB} 