#!/bin/bash
# ============================================================
# N26 预测推理
# 提交: sbatch --gpus=1 ./slurm/predict.sh
# 使用前修改 CKPT 指向训练产出的 .pkl
# ============================================================

#SBATCH --job-name=afm-predict
#SBATCH --output=/data02/%u/afm_protein/run/slurm_logs/predict-%j.out
#SBATCH --error=/data02/%u/afm_protein/run/slurm_logs/predict-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

PROJECT_DIR="$HOME/AFM_ML_code"
DATA_DIR="/data02/$USER/afm_protein"
VENV_DIR="$DATA_DIR/venv"
WORK_DIR="$DATA_DIR/run"

# ---- 修改这里 ----
CKPT="$WORK_DIR/outputs/YYYYMMDD-HHMMSS-protein/PROTEIN_E050_LX.XXXe-01.pkl"
AFM_DIR="$DATA_DIR/dataset/protein_train/afm"

mkdir -p "$WORK_DIR/slurm_logs"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "错误: 虚拟环境不存在: $VENV_DIR"
    exit 1
fi

cd "$PROJECT_DIR/code"

echo "Node: $(hostname) | CKPT: $CKPT | Start: $(date)"

python src/predict_protein.py \
    --ckpt "$CKPT" \
    --afm-dir "$AFM_DIR" \
    --save-xyz \
    --outdir "$WORK_DIR/predictions/" \
    --device cuda

echo "Done: $(date)"
