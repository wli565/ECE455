#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR
module load gcc/14.3.0
g++ -std=c++20 condition_task.cpp -o condition_task -I /srv/home/wli565/ECE455/HW08/taskflow/ -pthread
./condition_task
