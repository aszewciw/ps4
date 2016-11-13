
### Author: Mohammed A. Al Farhan, PhD student at ECRC, KAUST
### Email:  mohammed.farhan@kaust.edu.sa
### 
### ps4.c Makefile

all: clean ps4

.PHONY: clean

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CC  = ${CLINKER}
SRC = ps4.c
OBJ = ${SRC:.c=.o}

ps4: ${OBJ} chkopts
	${CC} -o ps4.out ${OBJ} ${PETSC_LIB}
	${RM} *.o

clean::
	${RM} -rf *.o *.out
