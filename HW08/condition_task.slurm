#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR
module load gcc
g++ -std=c++03 condition_task.cpp -o condition_task -I /srv/home/wli565/ECE455/HW08/ -pthread
./condition_task
