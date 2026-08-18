"""
从 PDB 文件生成机器训练用的 10 通道 AFM 灰度图 + XYZ 原子标签。

物理模型
--------
1. 提取蛋白质重原子 (C, N, O, S)，跳过 H 和 alternate 构象。
2. 将原子坐标各向异性归一化到物理盒子 (默认 25x25x8 Å) 内，
   保证 AFM 图像与标签使用完全一致的尺度 —— 修复旧版"标签 z 坐标
   越界被 clip 压扁"的缺陷。
3. 用范德华表面模型计算 AFM 高度图 (球形针尖 + 范德华半径)：
       H(x,y) = max_a { z_a + sqrt((R_tip + r_vdw_a)^2 - d_xy^2) - R_tip }
   默认 R_tip=0，即理想细针尖 -> 范德华表面，不会膨胀高度图。
4. 恒定高度模式生成 10 个通道：探针在 10 个高度 z 下探测，
   每个通道 = sigmoid((height_map - z) / sigma)，编码深度信息。

输出结构 (与 DetectDataset 'afm+label' 模式兼容)
------------------------------------------------
  out_dir/
    afm/{pdb_name}_{angle:03d}/0.png ~ 9.png
    label/{pdb_name}_{angle:03d}.xyz

用法
----
    python tools/pdb_to_afm.py \
        --pdb-dir dataset/protein_pdbs \
        --out-dir dataset/protein_train \
        --num-orientations 36
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# 兼容 Windows GBK 控制台: 让 stdout 能输出中文/特殊字符而不崩溃
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 范德华半径 (Å)，用于计算原子球表面
VAN_DER_WAALS = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}

# 参与训练检测的元素 (与 config protein_detect.py 的 ion_type 一致)
ELEMENTS = ("C", "N", "O")


def parse_pdb(pdb_path):
    """解析 PDB 文件，返回重原子列表。

    每个原子为 dict: {element, x, y, z, r_vdw}。
    - 只取第一个 MODEL (NMR 结构常含多个构象)。
    - 只取 ATOM 记录，排除 HETATM (水/配体/离子等非蛋白原子)。
    - 跳过氢原子 (AFM 中氢信号极弱)。
    - 跳过 alternate 构象 (高分辨率 PDB 的 A/B 双位置)。
    """
    atoms = []
    in_model = 0  # 0=无 MODEL 标签; n>=1=当前 MODEL 序号

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = line[0:6].strip()

            if record == "MODEL":
                parts = line.split()
                in_model = int(parts[1]) if len(parts) > 1 else 1
                continue
            if record == "ENDMDL":
                if in_model >= 1:
                    break  # 只保留第一个 MODEL
                continue

            # 只取第一个 MODEL (或无 MODEL 的单结构)
            if in_model > 1:
                continue

            # 只取蛋白质 ATOM 记录，排除水/配体等 HETATM
            if record != "ATOM":
                continue

            alt_loc = line[16:17].strip()
            if alt_loc not in ("", "A"):
                continue

            element = line[76:78].strip().upper()
            # 元素列可能为空，回退到原子名首字母 (CA->C, N->N, O->O)
            if not element:
                name = line[12:16].strip()
                if name and name[0].isalpha():
                    element = name[0].upper()

            if element not in VAN_DER_WAALS:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            atoms.append({
                "element": element,
                "x": x,
                "y": y,
                "z": z,
                "r_vdw": VAN_DER_WAALS[element],
            })

    return atoms


def rotate_z(atoms, angle_deg):
    """绕 z 轴旋转原子坐标 (用于多角度数据增广)。"""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for a in atoms:
        x, y = a["x"], a["y"]
        a["x"] = x * cos_t - y * sin_t
        a["y"] = x * sin_t + y * cos_t
    return atoms


def normalize_atoms(atoms, box, padding=0.08):
    """将原子坐标各向异性归一化到物理盒子内 (留边距)。

    每个轴独立缩放到 [pad, box - pad]，保证所有坐标严格落在盒子内，
    训练时不会被 vec2box 的 clip 压扁。返回每个轴的缩放因子用于诊断。
    """
    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    zs = [a["z"] for a in atoms]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    px = box[0] * padding
    py = box[1] * padding
    pz = box[2] * padding

    span_x = (xmax - xmin) or 1.0
    span_y = (ymax - ymin) or 1.0
    span_z = (zmax - zmin) or 1.0

    for a in atoms:
        a["x"] = px + (a["x"] - xmin) / span_x * (box[0] - 2 * px)
        a["y"] = py + (a["y"] - ymin) / span_y * (box[1] - 2 * py)
        a["z"] = pz + (a["z"] - zmin) / span_z * (box[2] - 2 * pz)

    scales = {
        "x": (xmax - xmin) / (box[0] - 2 * px),
        "y": (ymax - ymin) / (box[1] - 2 * py),
        "z": (zmax - zmin) / (box[2] - 2 * pz),
    }
    return atoms, scales


def compute_height_map(atoms, grid_size, box, tip_radius=0.0):
    """计算 AFM 高度图 (球形针尖 + 范德华半径)。

    H(x,y) = max_a { z_a + sqrt((R_tip + r_vdw_a)^2 - d_xy^2) - R_tip }
    d_xy^2 = (x - x_a)^2 + (y - y_a)^2，仅当 d_xy < R_tip + r_vdw_a。

    tip_radius=0 时退化为范德华表面 (理想细针尖)，不会膨胀高度图。
    """
    H, W = grid_size
    Lx, Ly = box[0], box[1]

    x_grid = np.linspace(0, Lx, H)
    y_grid = np.linspace(0, Ly, W)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    height_map = np.zeros((H, W), dtype=np.float64)

    for a in atoms:
        r_eff = tip_radius + a["r_vdw"]
        r_eff_sq = r_eff ** 2
        dx = X - a["x"]
        dy = Y - a["y"]
        d_sq = dx ** 2 + dy ** 2
        mask = d_sq < r_eff_sq
        if not mask.any():
            continue
        z_contact = a["z"] + np.sqrt(np.maximum(r_eff_sq - d_sq[mask], 0.0)) - tip_radius
        height_map[mask] = np.maximum(height_map[mask], z_contact)

    return height_map


def generate_afm_channels(height_map, num_channels=10, sigma=0.5):
    """从高度图生成多通道 AFM 灰度图 (恒定高度模式)。

    探针在 num_channels 个高度 z 下探测，从高到低均匀覆盖高度图范围。
    每个通道 = sigmoid((height_map - z) / sigma)：
      高度图高于探针 -> 接近 1 (亮，蛋白可见)
      高度图低于探针 -> 接近 0 (暗，蛋白不可见)

    返回 shape (num_channels, H, W)，值域 [0, 1]。
    """
    z_max = height_map.max()
    z_min = height_map.min()
    z_levels = np.linspace(z_max, z_min, num_channels)

    channels = []
    for z in z_levels:
        img = 1.0 / (1.0 + np.exp(-(height_map - z) / sigma))
        channels.append(img)

    return np.stack(channels, axis=0)


def save_afm_images(afm_channels, out_dir):
    """保存 AFM 通道为 PNG (0.png ~ N.png)。

    保存方向与 DetectDataset 读取时的 ROTATE_270 相抵消，
    保证像素坐标与标签 xy 坐标对齐。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(afm_channels.shape[0]):
        img_data = (np.clip(afm_channels[i], 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(img_data.T)
        img = img.rotate(90, expand=True)
        img.save(out_dir / f"{i}.png")


def save_atoms_xyz(atoms, out_path, box):
    """保存原子为 XYZ 标签 (ASE extxyz 格式)。

    坐标已归一化到盒子内，Lattice 与 config real_size 一致。
    """
    lines = [str(len(atoms))]
    lattice = (f"{box[0]:.4f} 0.0 0.0 0.0 {box[1]:.4f} 0.0 0.0 0.0 {box[2]:.4f}")
    lines.append(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3')

    for a in atoms:
        lines.append(f"{a['element']} {a['x']:.6f} {a['y']:.6f} {a['z']:.6f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_pdb(pdb_path, out_dir, num_orientations, num_channels,
                box, grid_size, sigma, tip_radius, padding):
    """处理单个 PDB：多角度生成 AFM 图像和 XYZ 标签，返回样本数。"""
    pdb_name = Path(pdb_path).stem
    afm_root = Path(out_dir) / "afm"
    label_root = Path(out_dir) / "label"
    afm_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    raw_atoms = parse_pdb(pdb_path)
    if not raw_atoms:
        print(f"  [SKIP] {pdb_name}: 未解析到重原子")
        return 0

    elem_counts = {}
    for a in raw_atoms:
        elem_counts[a["element"]] = elem_counts.get(a["element"], 0) + 1
    print(f"  {pdb_name}: {len(raw_atoms)} 个重原子 (分布: {elem_counts})")

    # 异常结构提示: 单蛋白通常 < 2000 原子，超过可能是多聚体/纤维/组装体
    if len(raw_atoms) > 2000:
        print(f"  [WARN] {pdb_name}: 原子数 {len(raw_atoms)} 异常偏大，"
              f"可能是多聚体/纤维结构，请确认是否适合作为单个蛋白的训练样本")

    angles = np.linspace(0, 360, num_orientations, endpoint=False)
    count = 0

    for angle in angles:
        atoms = [a.copy() for a in raw_atoms]
        rotate_z(atoms, angle)
        atoms, scales = normalize_atoms(atoms, box, padding=padding)

        height_map = compute_height_map(atoms, grid_size, box, tip_radius=tip_radius)
        afm_channels = generate_afm_channels(
            height_map, num_channels=num_channels, sigma=sigma)

        sample_name = f"{pdb_name}_{int(angle):03d}"
        save_afm_images(afm_channels, afm_root / sample_name)
        save_atoms_xyz(atoms, label_root / f"{sample_name}.xyz", box)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="PDB -> 10 通道 AFM 灰度图 + XYZ 标签")
    parser.add_argument("--pdb-dir", type=str, default="dataset/protein_pdbs",
                        help="PDB 文件目录")
    parser.add_argument("--out-dir", type=str, default="dataset/protein_train",
                        help="输出目录 (afm/ 与 label/)")
    parser.add_argument("--num-orientations", type=int, default=36,
                        help="每个蛋白的旋转角度数 (数据增广)")
    parser.add_argument("--num-channels", type=int, default=10,
                        help="AFM 通道数 (探针高度数)")
    parser.add_argument("--box", type=float, nargs=3, default=[25.0, 25.0, 8.0],
                        help="物理盒子尺寸 Lx Ly Lz (Å)，需与 config real_size 一致")
    parser.add_argument("--grid", type=int, nargs=2, default=[100, 100],
                        help="高度图像素 (H, W)，需与 config image_size 一致")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="sigmoid 平滑参数 (Å)，越小通道越锐利")
    parser.add_argument("--tip-radius", type=float, default=0.0,
                        help="针尖半径 (Å)，0=理想细针尖 (范德华表面)")
    parser.add_argument("--padding", type=float, default=0.08,
                        help="归一化边距 (盒子尺寸的比例)")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    if not pdb_dir.is_dir():
        raise SystemExit(f"PDB 目录不存在: {pdb_dir}")

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise SystemExit(f"目录下没有 PDB 文件: {pdb_dir}")

    print("=" * 60)
    print(f"PDB 目录: {pdb_dir} ({len(pdb_files)} 个文件)")
    print(f"输出目录: {out_dir}")
    print(f"盒子: {args.box} 埃 | 像素: {args.grid} | 通道: {args.num_channels}")
    print(f"每蛋白角度数: {args.num_orientations} | 针尖半径: {args.tip_radius} 埃")
    print("=" * 60)

    total = 0
    for pdb_file in pdb_files:
        total += process_pdb(
            pdb_file, out_dir,
            num_orientations=args.num_orientations,
            num_channels=args.num_channels,
            box=tuple(args.box),
            grid_size=tuple(args.grid),
            sigma=args.sigma,
            tip_radius=args.tip_radius,
            padding=args.padding,
        )

    print("=" * 60)
    print(f"完成: 共生成 {total} 个样本")
    print(f"  AFM:  {out_dir}/afm/")
    print(f"  Label: {out_dir}/label/")
    print("=" * 60)


if __name__ == "__main__":
    main()
