#!/bin/bash
# ============================================================
# N26 集群数据准备脚本 (在 login 节点运行)
# 运行: bash slurm/setup_data.sh
#
# 功能: 下载 PDB -> 生成 10 通道 AFM 训练数据 (72 角度) -> 软链接
# 前置: bash slurm/setup_venv.sh  (需先装好 venv 与依赖)
# ============================================================

set -e

PROJECT_DIR="${1:-$HOME/run/test/Radial-Train}"
BASE_DIR="$HOME/run/protein_afm"
VENV_DIR="$BASE_DIR/venv"

echo "============================================"
echo "N26 数据准备"
echo "  代码: $PROJECT_DIR"
echo "  数据: $BASE_DIR"
echo "============================================"

# ---- 激活环境 ----
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "错误: 虚拟环境不存在: $VENV_DIR"
    echo "请先运行: bash slurm/setup_venv.sh"
    exit 1
fi

# ---- 创建目录 ----
mkdir -p "$BASE_DIR/dataset/protein_pdbs"
mkdir -p "$BASE_DIR/dataset/protein_train"
mkdir -p "$BASE_DIR/slurm_logs"
mkdir -p "$PROJECT_DIR/code/slurm_logs"

# ---- 步骤 1: 下载 PDB (login 节点有网络) ----
echo ""
echo ">>> 步骤 1: 下载 PDB ..."
cd "$PROJECT_DIR/code"
python tools/download_pdbs.py \
    --outdir "$BASE_DIR/dataset/protein_pdbs"
echo "    PDB: $(ls "$BASE_DIR/dataset/protein_pdbs"/*.pdb 2>/dev/null | wc -l) 个"

# ---- 步骤 2: 生成 10 通道 AFM 训练数据 ----
echo ""
echo ">>> 步骤 2: 生成 AFM 训练数据 (每蛋白 72 角度) ..."
echo "    清空旧训练数据 ..."
rm -rf "$BASE_DIR/dataset/protein_train/afm" \
       "$BASE_DIR/dataset/protein_train/label"

python tools/pdb_to_afm.py \
    --pdb-dir "$BASE_DIR/dataset/protein_pdbs" \
    --out-dir "$BASE_DIR/dataset/protein_train" \
    --num-orientations 72 \
    --grid 512 512 \
    --tip-radius 2.0

echo "    AFM 样本: $(ls "$BASE_DIR/dataset/protein_train/afm" 2>/dev/null | wc -l) 个"
echo "    Label:    $(ls "$BASE_DIR/dataset/protein_train/label"/*.xyz 2>/dev/null | wc -l) 个"

# ---- 步骤 3: 软链接 ----
echo ""
echo ">>> 步骤 3: 创建软链接 ..."
ln -sfn "$BASE_DIR/dataset"  "$PROJECT_DIR/code/dataset"
ln -sfn "$BASE_DIR"           "$PROJECT_DIR/code/run_data"

echo ""
echo "============================================"
echo "数据准备完成!"
echo "下一步: sbatch --gpus=1 ./slurm/run.sh"
echo "============================================"
