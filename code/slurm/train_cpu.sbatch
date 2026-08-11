#!/bin/bash
#SBATCH --job-name=protein-afm-train
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --partition=cpu

# ============================================================
# 蛋白质 AFM 3D U-Net CPU 训练 (28 核)
#
# 提交方式: sbatch slurm/train_cpu.sbatch
# 查看队列: squeue -u $USER
# 查看输出: tail -f slurm_logs/protein-afm-train-*.out
# ============================================================

set -e

# ---- 环境配置 (请按集群实际路径修改) ----
PROJECT_DIR="$HOME/protein_afm/AFM_ML_code"
CONDA_ENV="protein_afm"

# ---- 激活环境 ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---- 切换到项目目录 ----
cd "$PROJECT_DIR/code"

# ---- 创建日志目录 ----
mkdir -p slurm_logs

# ---- 打印配置信息 ----
echo "============================================"
echo "节点: $(hostname)"
echo "CPU 核心: $SLURM_CPUS_PER_TASK"
echo "内存: ${SLURM_MEM_PER_NODE:-$SLURM_MEM_PER_CPU}"
echo "开始时间: $(date)"
echo "============================================"

# ---- 训练 ----
python src/train_protein.py \
    --device cpu \
    --num-threads "$SLURM_CPUS_PER_TASK" \
    --outdir outputs/ \
    --train-ratio 0.8

echo "============================================"
echo "完成时间: $(date)"
echo "============================================"
