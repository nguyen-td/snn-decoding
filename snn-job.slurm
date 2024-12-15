#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede3 SPR nodes
#
#   *** MPI Job in SPR Queue ***
# 
# Last revised: 23 April 2024
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch spr.mpi.slurm" on Stampede3 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do not use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per SPR node: 112
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      on more nodes using fewer tasks and/or threads 
#      per node to give each task access to more memory.
#
#   -- Don't worry about task layout.  By default, ibrun
#      will provide proper affinity and pinning.
#
#   -- You should always run out of $SCRATCH.  Your input
#      files, output files, and exectuable should be 
#      in the $SCRATCH directory hierarchy.
#
#----------------------------------------------------

#SBATCH -J snn-job           # Job name
#SBATCH -o snn-job.o%j       # Name of stdout output file
#SBATCH -e snn-job.e%j       # Name of stderr error file
#SBATCH -p spr               # Queue (partition) name
#SBATCH -N 4                 # Total # of nodes 
#SBATCH -t 6:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-user=nguyentd@tacc.utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...
module load python


# Always run your jobs out of $SCRATCH.  Your input files, output files, 
# and exectuable should be in the $SCRATCH directory hierarchy.  
# Change directories to your $SCRATCH directory where your executable is

cd $SCRATCH
cd snn-decoding

# Launch MPI code... 

python main.py