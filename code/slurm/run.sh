#!/bin/bash
# ============================================================
# N26 集群训练入口脚本
#
# 提交方式: sbatch --gpus=1 ./slurm/run.sh
# 多卡:     sbatch --gpus=2 ./slurm/run.sh
#
# 固定配比: 1 GPU = 10 CPU 核 + 38 GB 内存
#           2 GPU = 20 CPU 核 + 76 GB 内存
# ============================================================

#SBATCH --job-name=protein-afm
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

# ---- 集群路径 (按实际存放位置修改) ----
PROJECT_DIR="$HOME/AFM_ML_code"       # home 只放代码, <1GB
DATA_DIR="/data02/$USER/afm_protein"  # 数据集放 /data02
CONDA_ENV="protein_afm"

# ---- 激活环境 ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---- 切换到 run 目录 ----
WORK_DIR="$DATA_DIR/run"
mkdir -p "$WORK_DIR/slurm_logs"
cd "$PROJECT_DIR/code"

echo "============================================"
echo "节点: $(hostname)"
echo "GPU 卡数: $(nvidia-smi -L | wc -l)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "数据: $DATA_DIR"
echo "开始: $(date)"
echo "============================================"

# ---- 训练 ----
# N26 集群 1 GPU = 10 核, 使用 --num-workers 4 充分利用
python src/train_protein.py \
    --device cuda \
    --gpu-id 0 \
    --num-threads 6 \
    --num-workers 4 \
    --outdir "$WORK_DIR/outputs/" \
    --train-ratio 0.8

echo "============================================"
echo "完成: $(date)"
echo "============================================"
