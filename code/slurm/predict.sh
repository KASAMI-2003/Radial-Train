#!/bin/bash
# ============================================================
# N26 预测推理
# 提交: sbatch --gpus=1 ./slurm/predict.sh
# 使用前修改 CKPT 指向训练产出的 .pkl
# ============================================================

#SBATCH --job-name=afm-predict
#SBATCH --output=slurm_logs/predict-%j.out
#SBATCH --error=slurm_logs/predict-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

PROJECT_DIR="$HOME/AFM_ML_code"
BASE_DIR="$HOME/run/protein_afm"
VENV_DIR="$BASE_DIR/venv"

# ---- 修改这里为实际的 checkpoint 路径 ----
CKPT="$BASE_DIR/outputs/YYYYMMDD-HHMMSS-protein/PROTEIN_EXXX_LX.XXXe-01.pkl"
AFM_DIR="$BASE_DIR/dataset/protein_train/afm"

mkdir -p "$HOME/AFM_ML_code/code/slurm_logs"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR/code"

echo "Node: $(hostname) | CKPT: $CKPT | Start: $(date)"

python src/predict_protein.py \
    --ckpt "$CKPT" \
    --afm-dir "$AFM_DIR" \
    --save-xyz \
    --outdir "$BASE_DIR/predictions/" \
    --device cuda

echo "Done: $(date)"
