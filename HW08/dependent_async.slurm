#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=dependent_async.output

cd $SLURM_SUBMIT_DIR
module load gcc/14.3.0
g++ -std=c++20 dependent_async.cpp -o dependent_async -I /srv/home/wli565/ECE455/HW08 -pthread
./dependent_async
