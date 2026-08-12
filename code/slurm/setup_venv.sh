#!/bin/bash
# ============================================================
# N26 集群 Python 环境安装 (login 节点)
# 运行: bash slurm/setup_venv.sh
# ============================================================

set -e

PROJECT_DIR="${1:-$HOME/AFM_ML_code}"
BASE_DIR="$HOME/run/protein_afm"
VENV_DIR="$BASE_DIR/venv"

echo "============================================"
echo "N26 Python 环境安装 -> $VENV_DIR"
echo "============================================"

# ---- 检测 Python >= 3.8 ----
PYTHON_BIN=""
for py in python3.10 python3.9 python3.8 python3; do
    if command -v "$py" &>/dev/null; then
        PY_MAJOR=$("$py" -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$("$py" -c "import sys; print(sys.version_info.minor)")
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 8 ]; then
            PYTHON_BIN="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "错误: 未找到 Python >= 3.8"
    echo "可用 Python:"
    ls /usr/bin/python* 2>/dev/null
    exit 1
fi

echo "Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# ---- 创建 venv ----
mkdir -p "$BASE_DIR"
echo ">>> 创建 venv ..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# ---- PyTorch (CUDA 11.8, 兼容最广) ----
echo ">>> 安装 PyTorch CUDA ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ---- 其余依赖 ----
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
