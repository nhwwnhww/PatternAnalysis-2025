#!/bin/bash
#SBATCH --job-name=torch_test          # 作业名
#SBATCH --nodes=1                      # 节点数
#SBATCH --ntasks-per-node=1            # 每节点任务数
#SBATCH --cpus-per-task=1              # CPU 核心数
#SBATCH --gres=gpu:a100:1              # GPU 数量和型号
#SBATCH --mem=8G                       # 内存
#SBATCH --time=01:30:00                 # 最长运行时间
#SBATCH --output=torch_test.out         # 输出日志文件

# 加载模块
module load python/3.10
module load cuda/12.2
module load pytorch/2.2.0

# 激活 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# 启动 Python 脚本或 Notebook
# 方式1：运行 Python 脚本
# python test.py

# 方式2：运行 Jupyter Lab（注意: 需后续通过 SSH 隧道访问）
jupyter lab --no-browser --port=8888

