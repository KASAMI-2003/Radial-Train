"""
从 RCSB PDB 下载蛋白质结构文件
选取小型高分辨率蛋白作为训练数据集：
  - 3NIR: Crambin, 0.48Å, 46残基, ~5kDa
  - 2B97: Hydrophobin HFBII, 0.75Å, 140残基, ~14kDa
  - 1L2Y: Trp-cage, NMR, 20残基 (极小蛋白)
  - 1ENH: Engrailed homeodomain, NMR, 54残基
  - 1UBQ: Ubiquitin, 1.8Å, 76残基, ~8.5kDa
  - 2N9L: Villin headpiece, NMR, 35残基
  - 1PGB: Protein G B1 domain, NMR, 56残基
  - 1VII: Villin headpiece, NMR, 36残基
  - 5AWH: Insulin, 1.1Å, 51残基
  - 1YRF: BPTI mutant, 0.86Å, 58残基
"""

import os
import sys
import requests
import time
from pathlib import Path

# PDB 列表: (PDB_ID, 名称, 残基数, 分辨率)
PDB_LIST = [
    ("3nir", "crambin", 46, "0.48A"),
    ("2b97", "hydrophobin_hfbii", 140, "0.75A"),
    ("1ubq", "ubiquitin", 76, "1.80A"),
    ("5awh", "insulin", 51, "1.10A"),
    ("1yrf", "bpti_mutant", 58, "0.86A"),
    ("1l2y", "trp_cage", 20, "NMR"),
    ("2n9l", "villin_headpiece", 35, "NMR"),
    ("1pgb", "protein_g_b1", 56, "NMR"),
    ("1vii", "villin_hp36", 36, "NMR"),
    ("1enh", "engrailed_homeodomain", 54, "NMR"),
]


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    """从 RCSB PDB 下载 PDB 格式文件"""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    out_path = out_dir / f"{pdb_id.lower()}.pdb"

    if out_path.exists():
        print(f"  [SKIP] {pdb_id} 已存在: {out_path}")
        return out_path

    print(f"  下载 {pdb_id} ... ", end="", flush=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        out_path.write_text(resp.text, encoding="utf-8")
        size_kb = len(resp.text) / 1024
        print(f"OK ({size_kb:.1f} KB)")
        time.sleep(0.5)
        return out_path
    except requests.RequestException as e:
        print(f"FAILED: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载蛋白质 PDB 结构")
    parser.add_argument("--outdir", type=str, default=None,
                        help="输出目录 (默认: <repo_root>/dataset/protein_pdbs)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        out_dir = repo_root / "dataset" / "protein_pdbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {out_dir}")
    print(f"将下载 {len(PDB_LIST)} 个蛋白质结构\n")

    downloaded = []
    failed = []

    for pdb_id, name, n_res, resolution in PDB_LIST:
        print(f"[{resolution}] {name} ({pdb_id}, {n_res}残基)")
        result = download_pdb(pdb_id, out_dir)
        if result:
            downloaded.append((pdb_id, name, n_res, resolution, result))
        else:
            failed.append(pdb_id)

    print(f"\n{'='*60}")
    print(f"下载完成: {len(downloaded)} 成功, {len(failed)} 失败")
    if failed:
        print(f"失败列表: {', '.join(failed)}")
    print(f"\n文件保存在: {out_dir}")

    # 写入索引文件
    index_path = out_dir / "index.csv"
    with open(index_path, "w") as f:
        f.write("pdb_id,name,num_residues,resolution,filepath\n")
        for pdb_id, name, n_res, res, path in downloaded:
            f.write(f"{pdb_id},{name},{n_res},{res},{path.name}\n")
    print(f"索引文件: {index_path}")


if __name__ == "__main__":
    main()
