#!/bin/bash
# ============================================================
# N26 训练脚本
# 提交: sbatch --gpus=1 ./slurm/run.sh       (1卡=10核+38GB)
#       sbatch --gpus=2 ./slurm/run.sh       (2卡=20核+76GB)
# ============================================================

#SBATCH --job-name=protein-afm
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

PROJECT_DIR="$HOME/AFM_ML_code"
DATA_DIR="/data02/$USER/afm_protein"
VENV_DIR="$DATA_DIR/venv"
WORK_DIR="$DATA_DIR/run"

source "$VENV_DIR/bin/activate"
mkdir -p "$WORK_DIR/slurm_logs"
cd "$PROJECT_DIR/code"

echo "============================================"
echo "节点: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) x $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "开始: $(date)"
echo "============================================"

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
