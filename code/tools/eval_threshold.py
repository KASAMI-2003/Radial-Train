"""
阈值扫描评估脚本
用训练好的模型, 在测试集上扫描不同检测阈值, 找到使 AP/AR/F1 最优的阈值。

threshold 是后处理参数 (不参与训练), 因此无需重新训练。
当前训练代码里 threshold 硬编码为 0.5, 对类别极度不平衡的
原子检测任务来说偏高, 会导致大量正确预测被过滤 (AR 偏低)。

用法:
    python tools/eval_threshold.py \
        --ckpt <checkpoint.pkl> \
        --data dataset/protein_train \
        --device cuda --gpu-id 0
"""

import os
import sys
import argparse
import numpy as np
import torch

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.protein_detect import ProteinDetectConfig as Config
from src.network import UNetND
from src.dataset import DetectDataset
from src.utils import box2atom, ConfusionMatrix


def load_model(ckpt_path, device):
    cfg = Config()
    model = UNetND(**cfg.model.params.__dict__).to(device)
    params = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(params, strict=False)
    model.eval()
    model.requires_grad_(False)
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="threshold 扫描评估")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="模型 checkpoint .pkl 路径")
    parser.add_argument("--data", type=str, default="dataset/protein_train",
                        help="数据集目录 (含 afm/ 与 label/)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--thresholds", type=str,
                        default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5",
                        help="逗号分隔的阈值列表")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练/测试分割比例, 与训练一致")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="限制测试样本数 (0=全部)")
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    order = tuple(Config().dataset.ion_type)

    model, cfg = load_model(Path(args.ckpt), device)

    full_dts = DetectDataset(
        args.data, mode='afm+label',
        num_images=cfg.dataset.num_images,
        image_size=cfg.dataset.image_size,
        image_split=cfg.dataset.image_split,
        real_size=cfg.dataset.real_size,
        box_size=cfg.dataset.box_size,
        elements=(6, 7, 8),
        random_transform=False,
        normalize=True,
    )

    n_total = len(full_dts)
    n_train = int(n_total * args.train_ratio)
    n_test = n_total - n_train
    train_dts, test_dts = torch.utils.data.random_split(
        full_dts, [n_train, n_test],
        generator=torch.Generator().manual_seed(42))
    print(f"数据集: {n_total} 样本, 测试集 {len(test_dts)} 样本")

    if args.max_samples > 0:
        idxs = list(range(min(args.max_samples, len(test_dts))))
        test_dts = torch.utils.data.Subset(test_dts, idxs)
        print(f"仅评估前 {len(test_dts)} 个测试样本")

    # ---- 推理一遍, 缓存所有预测 ----
    print("推理中...")
    all_preds = []
    all_atoms = []
    with torch.no_grad():
        for i, (fname, afm, grids, atoms) in enumerate(test_dts):
            afm_t = torch.as_tensor(afm, dtype=torch.float32).unsqueeze(0).to(device)
            preds = model(afm_t)
            all_preds.append(preds[0].detach().cpu().numpy())  # (X, Y, Z, C)
            all_atoms.append(atoms)
            if (i + 1) % 20 == 0:
                print(f"  已推理 {i + 1}/{len(test_dts)}")

    # ---- 对每个阈值评估 ----
    print(f"\n阈值扫描 (元素: {order}, match_distance=1.0):")
    print(f"{'thr':>6} | {'C  AP/AR/F1':>16} | {'N  AP/AR/F1':>16} | {'O  AP/AR/F1':>16}")

    best = {}
    for thr in thresholds:
        metrics = ConfusionMatrix(
            count_types=order,
            real_size=cfg.dataset.real_size,
            split=cfg.dataset.split,
            match_distance=1.0,
        )
        for pred, atoms in zip(all_preds, all_atoms):
            out_atoms = box2atom(
                pred, cell=cfg.dataset.real_size, threshold=thr,
                cutoff=(1.8, 1.6, 1.5, 1.5), nms=cfg.dataset.nms, order=order)
            metrics.update(out_atoms, atoms)

        M = metrics.compute()  # (n_types, n_split, 7)
        parts = []
        for e in range(len(order)):
            ap = M[e, :, 3].mean().item()
            ar = M[e, :, 4].mean().item()
            f1 = (2 * ap * ar / (ap + ar)) if (ap + ar) > 0 else 0
            parts.append(f"{ap:.2f}/{ar:.2f}/{f1:.2f}")
            best.setdefault(order[e], []).append((f1, thr, ap, ar))

        print(f"{thr:>6} | {parts[0]:>16} | {parts[1]:>16} | {parts[2]:>16}")

    print("\n各元素最优 F1 阈值:")
    for e in order:
        b = max(best[e], key=lambda x: x[0])
        print(f"  {e}: thr={b[1]:.2f}  AP={b[2]:.2f} AR={b[3]:.2f} F1={b[0]:.2f}")

    print("\n完成")


if __name__ == "__main__":
    main()
