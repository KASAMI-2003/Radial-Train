#!/bin/bash
# ============================================================
# N26 集群数据准备脚本 (在 login 节点运行!)
#
# 运行方式: bash slurm/setup_data.sh
#
# 说明:
#   计算节点无网络，PDB 下载必须在 login 节点完成。
#   AFM 图像生成是纯 CPU 计算，也在 login 节点完成。
#   数据集存入 /data02 (home 仅 1GB, 不够用)。
# ============================================================

set -e

# ---- 配置 ----
PROJECT_DIR="${1:-$HOME/AFM_ML_code}"
DATA_DIR="/data02/$USER/afm_protein"
CONDA_ENV="protein_afm"

echo "============================================"
echo "N26 数据准备"
echo "  代码目录: $PROJECT_DIR"
echo "  数据目录: $DATA_DIR"
echo "============================================"

# ---- 激活环境 ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---- 创建数据目录 ----
mkdir -p "$DATA_DIR/dataset/protein_pdbs"
mkdir -p "$DATA_DIR/dataset/protein_train"
mkdir -p "$DATA_DIR/run"

# ---- 步骤 1: 下载 PDB ----
echo ""
echo ">>> 步骤 1: 下载 PDB 结构 (login 节点有网络)..."

cd "$PROJECT_DIR/code"

# 下载到 /data02 下的路径
python tools/download_pdbs.py \
    --outdir "$DATA_DIR/dataset/protein_pdbs"

echo "    完成: $(ls "$DATA_DIR/dataset/protein_pdbs"/*.pdb | wc -l) 个 PDB 文件"

# ---- 步骤 2: 生成 AFM 训练数据 ----
echo ""
echo ">>> 步骤 2: 生成 AFM 图像与 XYZ 标签..."

python tools/protein_afm_sim.py \
    --pdb-dir "$DATA_DIR/dataset/protein_pdbs" \
    --out-dir "$DATA_DIR/dataset/protein_train" \
    --num-orientations 36

echo "    完成: $(ls -d "$DATA_DIR/dataset/protein_train/afm"/*/ 2>/dev/null | wc -l) 个样本"

# ---- 步骤 3: 在代码目录创建软链接 ----
echo ""
echo ">>> 步骤 3: 创建软链接到 /data02..."

ln -sfn "$DATA_DIR/dataset" "$PROJECT_DIR/code/dataset"
ln -sfn "$DATA_DIR/run" "$PROJECT_DIR/code/run"

# ---- 更新 train_protein.py 配置中的路径 ----
echo ""
echo ">>> 步骤 4: 更新数据集路径..."

CONFIG_FILE="$PROJECT_DIR/code/configs/protein_detect.py"
sed -i "s|train_path: str = \".*\"|train_path: str = \"dataset/protein_train\"|" "$CONFIG_FILE"

echo ""
echo "============================================"
echo "数据准备完成!"
echo ""
echo "下一步:"
echo "  sbatch --gpus=1 ./slurm/run.sh"
echo "============================================"
