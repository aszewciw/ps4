
## Author:  Mohammed A. Al Farhan, PhD student at ECRC, KAUST
## Email:   mohammed.farhan@kaust.edu.sa
##
## A sample BASH script to execute a sample test case

#!/bin/bash

tol=$1

if ! [ -f ps4.out ]
then
  echo "The executable does not exist"
  echo "To build the executable type: make all"
else
  for((i = 0; i < 3; i+=1))
  do
    ./ps4.out -da_refine $i -ksp_monitor
    # ./ps4.out -da_refine $i -ksp_rtol $tol
  done
fi
