#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=static_tasking.output

cd $SLURM_SUBMIT_DIR

module load gcc/14.3.0

# Include path points to the folder that directly contains "taskflow/"
g++ -std=c++20 static_tasking.cpp -o static_tasking \
    -I $SLURM_SUBMIT_DIR \
    -pthread

stdbuf -o0 ./static_tasking
