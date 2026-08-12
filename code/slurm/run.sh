#!/bin/bash
# ============================================================
# N26 训练脚本
# 提交: sbatch --gpus=1 ./slurm/run.sh       (1卡=10核+38GB)
#       sbatch --gpus=2 ./slurm/run.sh       (2卡=20核+76GB)
# ============================================================

#SBATCH --job-name=protein-afm
#SBATCH --output=/data02/home/%u/run/test/Radial-Train/code/slurm_logs/train-%j.out
#SBATCH --error=/data02/home/%u/run/test/Radial-Train/code/slurm_logs/train-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

PROJECT_DIR="$HOME/run/test/Radial-Train"
BASE_DIR="$HOME/run/protein_afm"
VENV_DIR="$BASE_DIR/venv"

mkdir -p "$PROJECT_DIR/code/slurm_logs"
mkdir -p "$BASE_DIR/slurm_logs"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR/code"

echo "============================================"
echo "节点: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) x $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "数据: $BASE_DIR"
echo "开始: $(date)"
echo "============================================"

python src/train_protein.py \
    --device cuda \
    --gpu-id 0 \
    --num-threads 6 \
    --num-workers 4 \
    --outdir "$BASE_DIR/outputs/" \
    --train-ratio 0.8

echo "============================================"
echo "完成: $(date)"
echo "============================================"
