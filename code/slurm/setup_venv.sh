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
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VER"

mkdir -p "$BASE_DIR"

# ---- 创建 venv ----
echo ">>> 创建 venv ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 升级 pip (使用腾讯源)
pip install --upgrade pip setuptools wheel

# ---- PyTorch: 根据 Python 版本选择 ----
echo ">>> 安装 PyTorch ..."

# 避开腾讯镜像限制，直接用官方源
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PYTORCH_URL="https://download.pytorch.org/whl"

case "$PY_VER" in
    3.6|3.7)
        # PyTorch 1.13.1 最后支持 3.7, 3.6 用 1.10.2
        if [ "${PY_VER%.*}" = "3.6" ]; then
            echo "  Python 3.6 -> PyTorch 1.10.2+cu113"
            pip install "torch==1.10.2+cu113" "torchvision==0.11.3+cu113" "torchaudio==0.10.2+cu113" \
                -f "$PYTORCH_URL/cu113/torch_stable.html"
        else
            echo "  Python 3.7 -> PyTorch 1.13.1+cu116.2"
            pip install "torch==1.13.1+cu116.2" "torchvision==0.14.1+cu116.2" \
                -f "$PYTORCH_URL/cu116/torch_stable.html"
        fi
        ;;
    3.8)
        echo "  Python 3.8 -> torch 2.0.1+cu118"
        pip install torch==2.0.1 torchvision==0.15.2 --index-url "$PYTORCH_URL/cu118" -i "$MIRROR"
        ;;
    *)
        echo "  Python $PY_VER -> torch latest"
        pip install torch torchvision --index-url "$PYTORCH_URL/cu118" -i "$MIRROR"
        ;;
esac

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
