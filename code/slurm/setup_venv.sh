#!/bin/bash
# ============================================================
# N26 集群 Python 环境安装 (在 login 节点运行)
# 运行: bash slurm/setup_venv.sh
# ============================================================

set -e

PROJECT_DIR="${1:-$HOME/AFM_ML_code}"
BASE_DIR="$HOME/run/protein_afm"
VENV_DIR="$BASE_DIR/venv"

echo "============================================"
echo "N26 Python 环境安装 -> $VENV_DIR"
echo "============================================"

# ---- 检测 Python 版本 ----
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
echo "Python: $PY_MAJOR.$PY_MINOR"

mkdir -p "$BASE_DIR"

# ---- 创建 venv ----
echo ">>> 创建 venv ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# ---- PyTorch: 根据 Python 版本选择 ----
echo ">>> 安装 PyTorch ..."
PYTORCH_URL="https://download.pytorch.org/whl"

if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -eq 6 ]; then
    # Python 3.6: PyTorch 1.10.2 最后支持 3.6
    echo "  Python 3.6 -> PyTorch 1.10.2+cu102"
    pip install "torch==1.10.2+cu102" "torchvision==0.11.3+cu102" "torchaudio==0.10.2+cu102" \
        -f "$PYTORCH_URL/torch_stable.html"
elif [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -eq 7 ]; then
    # Python 3.7: PyTorch 1.13.1+cu117
    echo "  Python 3.7 -> PyTorch 1.13.1+cu117"
    pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" \
        -f "$PYTORCH_URL/torch_stable.html"
elif [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -eq 8 ]; then
    echo "  Python 3.8 -> PyTorch 2.0.1+cu118"
    pip install torch==2.0.1 torchvision==0.15.2 \
        --index-url "$PYTORCH_URL/cu118"
else
    echo "  Python $PY_MAJOR.$PY_MINOR -> PyTorch latest"
    pip install torch torchvision \
        --index-url "$PYTORCH_URL/cu118"
fi

# ---- 科学计算 ----
echo ">>> 安装依赖 ..."
pip install numpy scipy pillow scikit-learn pandas matplotlib ase torchmetrics

# ---- 验证 ----
echo ""
echo "============================================"
python -c "
import torch, numpy, ase, PIL
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  NumPy:   {numpy.__version__}')
print(f'  ASE:     {ase.__version__}')
"

echo ""
echo "完成! 下一步: bash slurm/setup_data.sh"
