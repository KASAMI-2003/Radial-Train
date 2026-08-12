#!/bin/bash
# ============================================================
# N26 训练脚本
# 提交: sbatch --gpus=1 ./slurm/run.sh       (1卡=10核+38GB)
#       sbatch --gpus=2 ./slurm/run.sh       (2卡=20核+76GB)
# ============================================================

#SBATCH --job-name=protein-afm
#SBATCH --output=/data02/%u/afm_protein/run/slurm_logs/train-%j.out
#SBATCH --error=/data02/%u/afm_protein/run/slurm_logs/train-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

PROJECT_DIR="$HOME/AFM_ML_code"
DATA_DIR="/data02/$USER/afm_protein"
VENV_DIR="$DATA_DIR/venv"
WORK_DIR="$DATA_DIR/run"

# 确保日志目录存在 (SBATCH 指令执行前不会运行脚本，这里为保险再建一次)
mkdir -p "$WORK_DIR/slurm_logs"
mkdir -p "$PROJECT_DIR/code/slurm_logs"

# 激活虚拟环境
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "错误: 虚拟环境不存在: $VENV_DIR"
    echo "请先在 login 节点运行: bash slurm/setup_venv.sh"
    exit 1
fi

cd "$PROJECT_DIR/code"

echo "============================================"
echo "节点: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) x $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "数据: $DATA_DIR"
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
