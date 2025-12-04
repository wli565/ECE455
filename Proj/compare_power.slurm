#!/usr/bin/env bash
#SBATCH --job-name=e455_varjoints        # job 名字
#SBATCH --partition=gpu                  # ⚠️ 这里改成 Euler 上的 GPU partition
#SBATCH --gres=gpu:1                     # 申请 1 块 GPU
#SBATCH --time=00:05:00                  # 最大运行时间
#SBATCH --cpus-per-task=4                # 给程序 4 个 CPU 线程
#SBATCH --output=varjoints.out        # 输出文件名，%j 是 job ID

# 切到提交脚本的目录
cd $SLURM_SUBMIT_DIR

# 载入 CUDA / NVML（按 Euler 文档来，示例）
module load cuda

# 编译（用你的 Makefile）
make compare_variable_joints

# 运行：参数 [num_matrices] [threads_per_block]
./compare_variable_joints 500000 64
