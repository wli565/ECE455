#!/usr/bin/env bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=instruction          # 先试 research，也可以改成 instruction
#SBATCH --gres=gpu:1                  # 要一块 GPU
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --output=test_gpu_%j.out

cd "$SLURM_SUBMIT_DIR"

# 看一眼 GPU 情况
nvidia-smi
