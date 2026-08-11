#!/bin/bash
#SBATCH --job-name=afm-gen
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu

# ============================================================
# 蛋白质 AFM 数据生成 (PDB -> AFM 图像 + XYZ 标签)
# 提交: sbatch slurm/gen_afm_data.sbatch
# ============================================================

set -e

PROJECT_DIR="$HOME/protein_afm/AFM_ML_code"
CONDA_ENV="protein_afm"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR/code"
mkdir -p slurm_logs

echo "============================================"
echo "节点: $(hostname) | 开始: $(date)"
echo "============================================"

echo ">>> 下载 PDB..."
python tools/download_pdbs.py

echo ">>> 生成 AFM 训练数据..."
python tools/protein_afm_sim.py \
    --pdb-dir dataset/protein_pdbs \
    --out-dir dataset/protein_train \
    --num-orientations 36

echo "============================================"
echo "完成: $(date)"
echo "============================================"
