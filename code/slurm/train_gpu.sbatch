#!/bin/bash
#SBATCH --job-name=protein-afm-gpu
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================
# 蛋白质 AFM 3D U-Net GPU 训练 (单卡)
# 提交: sbatch slurm/train_gpu.sbatch
# ============================================================

set -e

PROJECT_DIR="$HOME/protein_afm/AFM_ML_code"
CONDA_ENV="protein_afm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR/code"
mkdir -p slurm_logs

echo "============================================"
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "开始: $(date)"
echo "============================================"

python src/train_protein.py \
    --device cuda \
    --gpu-id 0 \
    --num-threads "$SLURM_CPUS_PER_TASK" \
    --outdir outputs/ \
    --train-ratio 0.8

echo "============================================"
echo "完成: $(date)"
echo "============================================"
