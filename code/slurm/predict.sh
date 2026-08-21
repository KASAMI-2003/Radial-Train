#!/bin/bash
# ============================================================
# N26 预测推理
# 提交: sbatch --gpus=1 ./slurm/predict.sh
# 自动使用最新训练的 checkpoint
# ============================================================

#SBATCH --job-name=afm-predict
#SBATCH --output=/data02/home/%u/run/test/Radial-Train/code/slurm_logs/predict-%j.out
#SBATCH --error=/data02/home/%u/run/test/Radial-Train/code/slurm_logs/predict-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

PROJECT_DIR="$HOME/run/test/Radial-Train"
BASE_DIR="$HOME/run/protein_afm"
VENV_DIR="$BASE_DIR/venv"

# ---- 自动找最新训练的 checkpoint ----
CKPT_DIR=$(ls -dt "$BASE_DIR"/outputs/*-protein 2>/dev/null | head -n 1)
CKPT=$(ls -t "$CKPT_DIR"/PROTEIN_E*.pkl 2>/dev/null | head -n 1)

if [ -z "$CKPT" ]; then
    echo "错误: 未找到 checkpoint, 请确认 $BASE_DIR/outputs/ 下有训练输出"
    exit 1
fi
AFM_DIR="$BASE_DIR/dataset/protein_train/afm"

mkdir -p "$PROJECT_DIR/code/slurm_logs"

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
