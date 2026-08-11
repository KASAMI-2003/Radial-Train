"""
蛋白质 AFM 模拟模块
基于 BioAFMviewer 的算法: 球形/锥形针尖与 Van der Waals 球体原子的非弹性碰撞

模拟过程:
  1. 将蛋白质放置于 z=0 平面
  2. 对每个 (x,y) 网格点，计算针尖首次接触任何原子时的高度
  3. 生成 10 个不同"虚拟探针高度"的 AFM 通道，编码深度信息

输出格式与原始冰代码的 AFM 图像格式兼容:
  - C x Z x H x W 的灰度图像堆叠
  - 每张 100x100 像素对应 25x25 Å 视场
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path


# 标准 Van der Waals 半径 (Å)
VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "H": 1.20,
    "P": 1.80,
    "CA": 1.70,
}


def extract_protein_atoms(pdb_path, keep_only=None):
    """从 PDB 文件中提取蛋白质原子坐标 (仅第一个模型)

    Args:
        pdb_path: PDB 文件路径
        keep_only: 保留的原子名称列表，如 ['CA', 'N', 'C', 'O']，None=全部重原子

    Returns:
        list[dict]: 原子信息列表
    """
    atoms = []
    seen_model = False
    with open(pdb_path, "r") as f:
        for line in f:
            # NMR 多模型: 只读取 MODEL 1
            if line.startswith("MODEL "):
                model_id = int(line[10:14].strip())
                if model_id > 1:
                    seen_model = True
                    continue
            if seen_model and line.startswith("ENDMDL"):
                break

            if line.startswith(("ATOM  ", "HETATM")):
                name = line[12:16].strip()
                element = line[76:78].strip()
                if not element:
                    element = name[0] if name else "C"

                if keep_only is not None and name not in keep_only:
                    continue

                if element == "H" and keep_only is None:
                    continue

                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue

                resname = line[17:20].strip()
                resid = int(line[22:26])
                chain = line[21:22].strip()
                r_vdw = VDW_RADII.get(element, 1.70)

                atoms.append({
                    "name": name,
                    "element": element,
                    "x": x, "y": y, "z": z,
                    "resname": resname,
                    "resid": resid,
                    "chain": chain,
                    "r_vdw": r_vdw,
                })

    return atoms


def center_protein(atoms):
    """将蛋白质质心移至 xy 平面中心，底部置于 z=0"""
    xs = np.array([a["x"] for a in atoms])
    ys = np.array([a["y"] for a in atoms])
    zs = np.array([a["z"] for a in atoms])

    cx, cy = xs.mean(), ys.mean()
    z_min = zs.min()

    for a in atoms:
        a["x"] -= cx
        a["y"] -= cy
        a["z"] -= z_min

    return atoms


def rotate_protein(atoms, angle_deg, axis="z"):
    """绕指定轴旋转蛋白质原子坐标"""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if axis == "z":
        for a in atoms:
            x, y = a["x"], a["y"]
            a["x"] = x * cos_t - y * sin_t
            a["y"] = x * sin_t + y * cos_t
    elif axis == "y":
        for a in atoms:
            x, z = a["x"], a["z"]
            a["x"] = x * cos_t - z * sin_t
            a["z"] = x * sin_t + z * cos_t

    return atoms


def compute_height_map(atoms, grid_size, physical_size, tip_radius=20.0):
    """使用球形针尖模型计算 AFM 高度图

    H(x,y) = max_{原子 a} { z_a + sqrt((r_tip + r_vdw)^2 - d_xy^2) }
    其中 d_xy^2 = (x - x_a)^2 + (y - y_a)^2 < (r_tip + r_vdw)^2

    Args:
        atoms: 原子列表 (已居中)
        grid_size: (H, W) 像素数
        physical_size: (Lx, Ly) 实际尺寸 (Å)
        tip_radius: 针尖半径 (Å)，HS-AFM 典型值 ~20Å

    Returns:
        height_map: (H, W) 高度图 (Å)
    """
    H, W = grid_size
    Lx, Ly = physical_size

    x_grid = np.linspace(0, Lx, H)
    y_grid = np.linspace(0, Ly, W)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    height_map = np.full((H, W), -np.inf)

    for a in atoms:
        dx = X - a["x"]
        dy = Y - a["y"]
        d_xy_sq = dx**2 + dy**2

        r_eff = tip_radius + a["r_vdw"]
        r_eff_sq = r_eff**2

        mask = d_xy_sq < r_eff_sq
        if not mask.any():
            continue

        z_contact = a["z"] + np.sqrt(np.maximum(r_eff_sq - d_xy_sq[mask], 0))
        height_map[mask] = np.maximum(height_map[mask], z_contact)

    height_map[~np.isfinite(height_map)] = 0.0

    return height_map


def generate_afm_channels(height_map, num_channels=10, z_range=None, sigma=1.0):
    """从高度图生成多通道 AFM 输入

    模拟恒定高度 AFM: 在不同探针高度下探测。
    每个通道编码了在该高度下蛋白质的"可见度"。
    多通道堆叠 = 不同 z 层的深度编码。

    Args:
        height_map: (H, W) 高度图 (Å)
        num_channels: 通道数
        z_range: (z_min, z_max) 探针高度范围
        sigma: sigmoid 平滑参数 (Å)

    Returns:
        channels: (num_channels, H, W) AFM 图像堆叠
    """
    if z_range is None:
        z_max = height_map.max()
        z_min = max(0, z_max - 20.0)
    else:
        z_min, z_max = z_range

    z_levels = np.linspace(z_max, z_min, num_channels)
    channels = []

    for z in z_levels:
        img = 1.0 / (1.0 + np.exp(-(height_map - z) / sigma))
        channels.append(img)

    return np.stack(channels, axis=0)


def add_experimental_noise(afm_channels, noise_level=0.03, blur_sigma=0.8):
    """添加模拟实验噪声

    Args:
        afm_channels: (Z, H, W) AFM 图像堆叠
        noise_level: 高斯噪声标准差
        blur_sigma: 高斯模糊 sigma

    Returns:
        noisy_channels: (Z, H, W)
    """
    noisy = afm_channels.copy()
    for i in range(noisy.shape[0]):
        noisy[i] = gaussian_filter(noisy[i], sigma=blur_sigma)
        noisy[i] += np.random.randn(*noisy[i].shape) * noise_level
        noisy[i] = np.clip(noisy[i], 0, 1)
    return noisy


def atoms_to_voxel_labels(atoms, grid_size, physical_size, element_types):
    """将原子坐标转换为 3D voxel 标签

    输出格式与原始代码兼容:
      voxels[ix, iy, iz, ch:ch+4] = [confidence, dx, dy, dz]

    Args:
        atoms: 原子列表
        grid_size: (Nx, Ny, Nz) voxel 数量
        physical_size: (Lx, Ly, Lz) 实际尺寸 (Å)
        element_types: 元素类型列表

    Returns:
        voxels: (Nx, Ny, Nz, 4*N_elem) numpy array
    """
    Nx, Ny, Nz = grid_size
    Lx, Ly, Lz = physical_size
    n_elems = len(element_types)

    voxels = np.zeros((Nx, Ny, Nz, 4 * n_elems), dtype=np.float32)

    for a in atoms:
        elem_idx = -1
        for ei, et in enumerate(element_types):
            if a["name"] == et or a["element"] == et:
                elem_idx = ei
                break
        if elem_idx < 0:
            continue

        fx = a["x"] / Lx
        fy = a["y"] / Ly
        fz = a["z"] / Lz

        if not (0 <= fx < 1 and 0 <= fy < 1 and 0 <= fz < 1):
            continue

        ix = min(int(fx * Nx), Nx - 1)
        iy = min(int(fy * Ny), Ny - 1)
        iz = min(int(fz * Nz), Nz - 1)

        ox = fx * Nx - ix
        oy = fy * Ny - iy
        oz = fz * Nz - iz

        ch = elem_idx * 4
        voxels[ix, iy, iz, ch + 0] = 1.0
        voxels[ix, iy, iz, ch + 1] = ox
        voxels[ix, iy, iz, ch + 2] = oy
        voxels[ix, iy, iz, ch + 3] = oz

    return voxels


def save_afm_images(afm_channels, out_dir):
    """保存 AFM 图像为 PNG + NPZ 文件 (与 DetectDataset 兼容)

    目录结构: out_dir/{0,1,...,9}.png + img.npz

    Args:
        afm_channels: (Z, H, W) 图像堆叠
        out_dir: 输出目录 (会被创建)
    """
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(afm_channels.shape[0]):
        img_data = (afm_channels[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_data.T)
        img = img.rotate(90, expand=True)
        img.save(out_dir / f"{i}.png")

    np.savez_compressed(out_dir / "img.npz",
                        img=(afm_channels * 255).astype(np.uint8))


def save_atoms_xyz(atoms, out_path, cell_diag):
    """保存原子为 XYZ 格式 (与 ASE / DetectDataset 兼容)

    格式:
      <num_atoms>
      Lattice="Lx 0 0 0 Ly 0 0 0 Lz" Properties=species:S:1:pos:R:3
      <elem> <x> <y> <z>
      ...

    Args:
        atoms: list[dict] 原子列表
        out_path: 输出文件路径
        cell_diag: (Lx, Ly, Lz) 晶胞对角线
    """
    lines = [str(len(atoms))]
    lattice = f"{cell_diag[0]:.4f} 0.0 0.0 0.0 {cell_diag[1]:.4f} 0.0 0.0 0.0 {cell_diag[2]:.4f}"
    lines.append(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3')

    for a in atoms:
        lines.append(f"{a['element']} {a['x']:.6f} {a['y']:.6f} {a['z']:.6f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def process_protein_for_training(
    pdb_path,
    out_dir,
    element_types=("CA", "C", "N", "O"),
    num_channels=10,
    image_size=(100, 100),
    physical_size=(25.0, 25.0, 8.0),
    box_size=(32, 32, 8),
    tip_radius=20.0,
    num_orientations=36,
    noise_level=0.03,
    blur_sigma=0.8,
):
    """处理单个蛋白质: 多方向生成 AFM 图像和 XYZ 标签

    输出结构 (与 DetectDataset 'afm+label' 模式兼容):
      out_dir/
        afm/
          {pdb_name}_{angle}/       ← 每个样本一个目录
            0.png, 1.png, ..., 9.png
            img.npz
        label/
          {pdb_name}_{angle}.xyz    ← 原子位置 XYZ 文件

    Returns:
        int: 生成的样本数
    """
    pdb_name = Path(pdb_path).stem
    afm_root = Path(out_dir) / "afm"
    label_root = Path(out_dir) / "label"
    afm_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    all_atoms = extract_protein_atoms(pdb_path, keep_only=element_types)
    if len(all_atoms) == 0:
        print(f"  警告: {pdb_name} 没有匹配的原子 ({element_types})")
        return 0

    # 计算元素分布用于统计
    elem_counts = {}
    for a in all_atoms:
        elem_counts[a["element"]] = elem_counts.get(a["element"], 0) + 1
    print(f"    原子数: {len(all_atoms)} (分布: {elem_counts})")

    angles = np.linspace(0, 360, num_orientations, endpoint=False)
    count = 0

    for angle in angles:
        atoms = [a.copy() for a in all_atoms]
        center_protein(atoms)
        rotate_protein(atoms, angle, axis="z")

        # 随机小角度倾斜 (模拟实验中的非完美朝向)
        tilt_angle = np.random.uniform(-15, 15)
        rotate_protein(atoms, tilt_angle, axis="y")

        xs = [a["x"] for a in atoms]
        ys = [a["y"] for a in atoms]
        zs = [a["z"] for a in atoms]

        # 跳过超出视场的样本
        if max(xs) - min(xs) > physical_size[0] * 1.1:
            continue

        # 在视场内随机平移
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        shift_x = np.random.uniform(-0.1, 0.1) * physical_size[0]
        shift_y = np.random.uniform(-0.1, 0.1) * physical_size[1]
        for a in atoms:
            a["x"] = a["x"] - cx + physical_size[0] / 2 + shift_x
            a["y"] = a["y"] - cy + physical_size[1] / 2 + shift_y

        # ---- 生成 AFM 图像 ----
        height_map = compute_height_map(
            atoms, grid_size=image_size,
            physical_size=physical_size[:2], tip_radius=tip_radius,
        )

        # z_range 从 height_map 的非零最小值到最大值，使 10 个通道
        # 均匀覆盖蛋白质的全部可见高度范围，提供充分深度编码。
        hm_nonzero = height_map[height_map > 0]
        z_min = hm_nonzero.min() if len(hm_nonzero) > 0 else 0
        z_max = height_map.max()
        afm_channels = generate_afm_channels(
            height_map, num_channels=num_channels,
            z_range=(z_min, z_max), sigma=0.5,
        )
        afm_channels = add_experimental_noise(
            afm_channels, noise_level=noise_level, blur_sigma=blur_sigma,
        )

        # ---- 保存 ----
        sample_name = f"{pdb_name}_{int(angle):03d}"

        # AFM: 每个样本一个目录
        save_afm_images(afm_channels, afm_root / sample_name)

        # Label: XYZ 文件 (DetectDataset 的 _read_label 可读取)
        save_atoms_xyz(atoms, label_root / f"{sample_name}.xyz",
                       cell_diag=physical_size)

        count += 1

    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="蛋白质 AFM 数据生成")
    parser.add_argument("--pdb-dir", type=str, default="dataset/protein_pdbs")
    parser.add_argument("--out-dir", type=str, default="dataset/protein_train")
    parser.add_argument("--num-orientations", type=int, default=36)
    parser.add_argument("--num-channels", type=int, default=10)
    parser.add_argument("--tip-radius", type=float, default=20.0)
    parser.add_argument("--elements", type=str, nargs="+",
                        default=["CA", "C", "N", "O"])
    parser.add_argument("--image-size", type=int, nargs=2, default=[100, 100])
    parser.add_argument("--physical-size", type=float, nargs=3,
                        default=[25.0, 25.0, 8.0])
    parser.add_argument("--box-size", type=int, nargs=3,
                        default=[32, 32, 8])

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pdb_dir = repo_root / args.pdb_dir
    out_dir = repo_root / args.out_dir

    if not pdb_dir.exists():
        print(f"错误: PDB 目录不存在: {pdb_dir}")
        print("请先运行: python tools/download_pdbs.py")
        return

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"错误: {pdb_dir} 中没有 .pdb 文件")
        return

    print(f"找到 {len(pdb_files)} 个 PDB 文件")
    print(f"元素: {args.elements} | 每蛋白 {args.num_orientations} 方向")
    print(f"输出: {out_dir}\n")

    total = 0
    for pdb_path in pdb_files:
        print(f"处理: {pdb_path.name}")
        n = process_protein_for_training(
            pdb_path=pdb_path, out_dir=out_dir,
            element_types=tuple(args.elements),
            num_channels=args.num_channels,
            image_size=tuple(args.image_size),
            physical_size=tuple(args.physical_size),
            box_size=tuple(args.box_size),
            tip_radius=args.tip_radius,
            num_orientations=args.num_orientations,
        )
        print(f"  -> {n} 样本")
        total += n

    print(f"\n总计: {total} 样本")


if __name__ == "__main__":
    main()
