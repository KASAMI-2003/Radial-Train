#!/bin/bash
#SBATCH --job-name=afm-predict
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================
# 蛋白质 AFM 预测 (GPU 推理)
# 提交: sbatch slurm/predict.sbatch
# 使用前修改 CKPT 指向最新 checkpoint
# ============================================================

set -e

PROJECT_DIR="$HOME/protein_afm/AFM_ML_code"
CONDA_ENV="protein_afm"
CKPT="outputs/YYYYMMDD-HHMMSS-protein/PROTEIN_E050_LX.XXXe-01.pkl"
AFM_DIR="dataset/protein_train/afm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR/code"
mkdir -p slurm_logs

echo "Node: $(hostname) | CKPT: $CKPT | Start: $(date)"

python src/predict_protein.py \
    --ckpt "$CKPT" \
    --afm-dir "$AFM_DIR" \
    --save-xyz \
    --outdir predictions/ \
    --device cuda

echo "Done: $(date)"
