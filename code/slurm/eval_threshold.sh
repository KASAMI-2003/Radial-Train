#!/bin/bash
# ============================================================
# N26 阈值扫描评估
# 提交: sbatch --gpus=1 ./slurm/eval_threshold.sh
# 说明: 登录节点无 GPU, 必须在计算节点运行
# ============================================================

#SBATCH --job-name=afm-eval
#SBATCH --output=/data02/home/%u/run/test/Radial-Train/code/slurm_logs/eval-%j.out
#SBATCH --error=/data02/home/%u/run/test/Radial-Train/code/slurm_logs/eval-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

PROJECT_DIR="$HOME/run/test/Radial-Train"
BASE_DIR="$HOME/run/protein_afm"
VENV_DIR="$BASE_DIR/venv"

# ---- 自动找最新训练的 checkpoint (最后一个保存的 = loss 最低) ----
CKPT_DIR="$BASE_DIR/outputs/20260818-153633-protein"
CKPT=$(ls -t "$CKPT_DIR"/PROTEIN_E*.pkl 2>/dev/null | head -n 1)

if [ -z "$CKPT" ]; then
    echo "错误: 未找到 checkpoint, 请确认目录 $CKPT_DIR 存在 .pkl 文件"
    exit 1
fi

mkdir -p "$PROJECT_DIR/code/slurm_logs"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR/code"

echo "============================================"
echo "节点: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "CKPT: $CKPT"
echo "开始: $(date)"
echo "============================================"

python tools/eval_threshold.py \
    --ckpt "$CKPT" \
    --data "$BASE_DIR/dataset/protein_train" \
    --device cuda --gpu-id 0

echo "============================================"
echo "完成: $(date)"
echo "============================================"
