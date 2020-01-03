#!/bin/bash 
#MSUB -l nodes=1:ppn=1
#MSUB -l walltime=00:01:00 
#MSUB -l pmem=6000mb 
#MSUB -m e 
#MSUB -M jan.griesser@imtek.uni-freiburg.de 
#MSUB -N EigenValVecs.N864kk.KA.Smallest
 
 # Load Libaries
module purge 
module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles/
module load mpi4py/3.0.0-python-3.6.5-openmpi-3.1-gnu-7.3
export PYTHONPATH="/home/fr/fr_fr/fr_jg1080/.local/lib/python3.6/site-packages:$PYTHONPATH"

# Jobs starts in submit directory, change if necessary 
cd $PBS_O_WORKDIR
 
python3 convert_petsc_matrix_to_npz.py -h 
