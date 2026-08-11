#!/bin/bash
# ============================================================
# N26 集群 Python 环境安装 (在 login 节点运行)
# 运行: bash slurm/setup_venv.sh
# ============================================================

set -e

PROJECT_DIR="${1:-$HOME/AFM_ML_code}"
DATA_DIR="/data02/$USER/afm_protein"
VENV_DIR="$DATA_DIR/venv"

echo "============================================"
echo "N26 Python 环境安装 -> $VENV_DIR"
echo "============================================"

# ---- 创建虚拟环境 ----
echo ">>> 创建 venv ..."
python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ---- PyTorch (CUDA 11.8) ----
echo ">>> 安装 PyTorch CUDA ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ---- 科学计算 ----
echo ">>> 安装依赖 ..."
pip install numpy scipy pillow scikit-learn pandas matplotlib ase torchmetrics

# ---- 验证 ----
echo ""
echo "============================================"
python -c "
import torch, numpy, ase, PIL
print(f'  PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')
print(f'  NumPy:   {numpy.__version__}')
print(f'  ASE:     {ase.__version__}')
"

echo ""
echo "完成! 下一步: bash slurm/setup_data.sh"
