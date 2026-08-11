#!/bin/bash
# ============================================================
# N26 集群预测推理脚本
#
# 提交方式: sbatch --gpus=1 ./slurm/predict.sh
#
# 使用前修改下面两行:
#   CKPT   -> 训练产出的 .pkl checkpoint 路径
#   AFM_DIR -> 待预测的 AFM 图像目录
# ============================================================

#SBATCH --job-name=afm-predict
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

# ---- 集群路径 (按实际存放位置修改) ----
PROJECT_DIR="$HOME/AFM_ML_code"
DATA_DIR="/data02/$USER/afm_protein"
CONDA_ENV="protein_afm"

# ---- 修改这里 ----
CKPT="$DATA_DIR/run/outputs/YYYYMMDD-HHMMSS-protein/PROTEIN_E050_LX.XXXe-01.pkl"
AFM_DIR="$DATA_DIR/dataset/protein_train/afm"

# ---- 激活环境 ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

WORK_DIR="$DATA_DIR/run"
mkdir -p "$WORK_DIR/slurm_logs"
cd "$PROJECT_DIR/code"

echo "Node: $(hostname) | CKPT: $CKPT | Start: $(date)"

python src/predict_protein.py \
    --ckpt "$CKPT" \
    --afm-dir "$AFM_DIR" \
    --save-xyz \
    --outdir "$WORK_DIR/predictions/" \
    --device cuda

echo "Done: $(date)"
