"""
蛋白质 AFM 预测脚本
加载训练好的模型，对 AFM 图像预测原子位置 (C, N, O)
支持 CPU / GPU，适配 Windows / Linux。

用法:
    python src/predict_protein.py --ckpt <checkpoint.pkl> --afm dataset/protein_train/afm/1enh_040
    python src/predict_protein.py --ckpt <checkpoint.pkl> --afm-dir dataset/protein_train/afm --save-xyz --device cuda
"""

import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from ase.io import write

# 兼容新旧 Pillow
try:
    ROTATE_270 = Image.Transpose.ROTATE_270
except AttributeError:
    ROTATE_270 = Image.ROTATE_270

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.protein_detect import ProteinDetectConfig as Config
from src.network import UNetND
from src.utils import box2atom

IS_WINDOWS = sys.platform == "win32"


def load_model(ckpt_path: Path, device: torch.device):
    """加载模型和配置"""
    cfg = Config()
    model = UNetND(**cfg.model.params.__dict__).to(device)
    params = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(params, strict=False)
    model.eval()
    model.requires_grad_(False)
    print(f"已加载模型: {ckpt_path}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  设备: {device}")
    return model, cfg


def load_afm(afm_dir: Path, image_size=(100, 100)) -> torch.Tensor:
    """从目录加载 AFM 图像堆叠 (10通道 × 100×100)，与训练读取方式完全一致"""
    if (afm_dir / "img.npz").exists():
        data = np.load(afm_dir / "img.npz")["img"].astype(np.float32) / 255.0
    else:
        pngs = sorted(
            [f for f in afm_dir.iterdir() if f.suffix == ".png" and f.stem.isdigit()],
            key=lambda x: int(x.stem),
        )
        channels = []
        for p in pngs:
            # 训练时 dataset._read_afm 会 transpose(ROTATE_270) + resize
            img = Image.open(p).convert("L").transpose(ROTATE_270)
            img = np.array(img.resize(image_size))
            channels.append(img)
        data = np.stack(channels, axis=0).astype(np.float32) / 255.0

    # 逐通道 min-max 归一化 (与训练 normalize_fn 一致)
    maxv = data.max(axis=(-2, -1), keepdims=True)
    minv = data.min(axis=(-2, -1), keepdims=True)
    data = (data - minv) / (maxv - minv + 1e-8)

    # 添加 batch 和 channel 维度: (Z, H, W) -> (1, 1, Z, H, W)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_atoms(model, afm_tensor, cfg, device):
    """预测原子位置列表"""
    inps = afm_tensor.to(device)
    preds = model(inps)                       # (1, X, Y, Z, C) 已 sigmoid
    preds = preds.detach().cpu().numpy()
    return preds


def voxel_to_atoms(preds, cfg):
    """将模型输出的 voxel 网格转换为 ASE Atoms 列表"""
    all_atoms = []
    for i in range(len(preds)):
        atoms = box2atom(
            preds[i],
            cell=cfg.dataset.real_size,
            threshold=cfg.dataset.threshold,
            cutoff=(1.8, 1.6, 1.5),         # C, N, O cutoff
            nms=cfg.dataset.nms,
            order=cfg.dataset.ion_type,
        )
        all_atoms.append(atoms)
    return all_atoms


def main():
    parser = argparse.ArgumentParser(description="蛋白质 AFM 预测")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="模型 checkpoint .pkl 路径")
    parser.add_argument("--afm", type=str, default=None,
                        help="单个 AFM 图像目录")
    parser.add_argument("--afm-dir", type=str, default=None,
                        help="批量 AFM 图像父目录")
    parser.add_argument("--outdir", type=str, default="predictions/",
                        help="输出目录")
    parser.add_argument("--save-xyz", action="store_true",
                        help="保存 XYZ 文件")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU 编号 (仅 --device cuda 时生效)")
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"错误: checkpoint 不存在: {ckpt}")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(ckpt, device)

    # 收集预测目标
    afm_dirs = []
    if args.afm:
        afm_dirs.append(Path(args.afm))
    if args.afm_dir:
        root = Path(args.afm_dir)
        for d in sorted(root.iterdir()):
            if d.is_dir():
                afm_dirs.append(d)

    if not afm_dirs:
        print("错误: 需要指定 --afm 或 --afm-dir")
        return

    print(f"预测 {len(afm_dirs)} 个样本...\n")

    for afm_dir in afm_dirs:
        if not afm_dir.exists():
            print(f"  跳过 (不存在): {afm_dir}")
            continue

        tensor = load_afm(afm_dir, cfg.dataset.image_size)
        preds = predict_atoms(model, tensor, cfg, device)
        atoms_list = voxel_to_atoms(preds, cfg)

        atoms = atoms_list[0]
        n_total = len(atoms)

        # 按元素统计
        counts = {}
        for at in atoms:
            sym = at.symbol
            counts[sym] = counts.get(sym, 0) + 1
        count_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

        name = afm_dir.name
        print(f"  {name}: 检测到 {n_total} 个原子 ({count_str})")

        if args.save_xyz:
            xyz_path = outdir / f"{name}.xyz"
            write(str(xyz_path), atoms)
            print(f"    已保存: {xyz_path}")

    print(f"\n预测完成! 输出目录: {outdir}")


if __name__ == "__main__":
    main()
