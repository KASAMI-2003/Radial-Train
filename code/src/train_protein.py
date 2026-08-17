"""
蛋白质 AFM 3D U-Net 训练脚本

基于冰代码的 train_det.py，适配蛋白质多元素预测。
自动检测 Windows/Linux 平台，适配分布式环境 (SLURM)。

用法:
    # CPU 训练 (自动检测核心数)
    python src/train_protein.py --device cpu --outdir outputs/

    # GPU 训练
    python src/train_protein.py --device cuda --outdir outputs/

    # 强制指定线程数 / GPU ID
    python src/train_protein.py --device cuda --gpu-id 0 --num-threads 8
"""

import os
import sys
import time
import shutil
import numpy as np
import torch
import utils
from functools import partial
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanMetric
from pathlib import Path
from multiprocessing import Pool

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.protein_detect import ProteinDetectConfig as Config
from src.network import UNetND
from src.dataset import DetectDataset
from src.utils import box2atom, plot_preditions


IS_WINDOWS = sys.platform == "win32"


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU 编号 (仅 --device cuda 时生效)")
    parser.add_argument("--outdir", type=str, default="outputs/",
                        help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Training split ratio")
    parser.add_argument("--num-threads", type=int, default=0,
                        help="CPU / OMP 线程数 (0=自动检测)")
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader workers (-1=自动)")
    return parser.parse_args()


class ProteinTrainer:
    def __init__(self, cfg: Config, train_ratio=0.8):
        self.cfg = cfg
        self.device = torch.device(self.cfg.setting.device)
        self.outdir = Path(self.cfg.setting.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log = utils.get_logger("protein-train", self.outdir)
        self.iters = 0
        self.epoch = 0

        self.log.info(f"设备: {self.device}, workers: {cfg.setting.num_workers}")
        self.log.info(f"元素类型: {cfg.dataset.ion_type}")
        self.log.info(f"物理尺寸: {cfg.dataset.real_size}")
        self.log.info(f"Voxel 网格: {cfg.dataset.box_size}")

        # ---- 创建模型 ----
        self.model = UNetND(**self.cfg.model.params.__dict__).to(self.device)
        self.tune_model = None  # 不使用 CycleGAN (没有实验数据)

        self.load_model()

        # ---- 数据集 ----
        # elements 需要原子序号: C=6, N=7, O=8
        full_dts = DetectDataset(
            cfg.dataset.train_path,
            mode='afm+label',
            num_images=cfg.dataset.num_images,
            image_size=cfg.dataset.image_size,
            image_split=cfg.dataset.image_split,
            real_size=self.cfg.dataset.real_size,
            box_size=self.cfg.dataset.box_size,
            elements=(6, 7, 8),  # C, N, O 原子序数
            random_transform=True,
            random_blur=2.0,
            random_cutout=True,
            random_jitter=True,
            random_noisy=0.1,
            random_shift=False,   # 禁用 pixel_shift (numpy roll 整数类型 bug)
            normalize=True,
        )

        # 分割训练/测试集
        n_total = len(full_dts)
        n_train = int(n_total * train_ratio)
        n_test = n_total - n_train
        self.train_dts, self.test_dts = random_split(
            full_dts, [n_train, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        self.log.info(f"数据集: {n_total} 样本 (训练={n_train}, 测试={n_test})")

        collate_fn = full_dts.collate_fn

        if self.cfg.setting.debug:
            n_dbg = min(100, len(self.train_dts))
            self.train_dts = torch.utils.data.Subset(self.train_dts, range(n_dbg))
            n_dbg = min(20, len(self.test_dts))
            self.test_dts = torch.utils.data.Subset(self.test_dts, range(n_dbg))

        self.train_dtl = DataLoader(
            self.train_dts,
            cfg.setting.batch_size, True,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory,
            collate_fn=collate_fn,
        )
        self.test_dtl = DataLoader(
            self.test_dts,
            cfg.setting.batch_size, False,
            num_workers=cfg.setting.num_workers,
            pin_memory=cfg.setting.pin_memory,
            collate_fn=collate_fn,
        )

        # ---- 优化器 ----
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, **self.cfg.scheduler.params.__dict__
        )

        # ---- 评估指标 ----
        self.atom_metrics = utils.MetricCollection(
            M=utils.ConfusionMatrix(
                count_types=cfg.dataset.ion_type,
                real_size=cfg.dataset.real_size,
                split=self.cfg.dataset.split,
                match_distance=1.0,
            )
        ).to(self.device)

        self.grid_metrics = utils.MetricCollection(
            loss=MeanMetric(),
            grad=MeanMetric(),
            conf=MeanMetric(),
            xy=MeanMetric(),
            z=MeanMetric(),
        ).to(self.device)

        self.log.info(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        self.save_paths = []
        self.best = np.inf

    def fit(self):
        for epoch in range(1, self.cfg.setting.epoch + 1):
            self.epoch = epoch
            epoch_start = time.time()

            gm = self.train_one_epoch()
            gm, atom_metric = self.test_one_epoch()
            loss = gm['loss']
            M = atom_metric['M']

            elapsed = (time.time() - epoch_start) / 60
            self.log.info(f"Epoch {epoch:3d}/{self.cfg.setting.epoch} | "
                          f"Loss {loss:.4e} | Time {elapsed:.1f}min | "
                          f"{'SAVED' if loss < self.best else '---'}")

            for elem_idx, elem_name in enumerate(self.cfg.dataset.ion_type):
                ap = M[elem_idx, :, 3].mean().item()
                ar = M[elem_idx, :, 4].mean().item()
                f1 = (2 * ap * ar / (ap + ar)) if (ap + ar) > 0 else 0
                self.log.info(f"  {elem_name}: AP={ap:.2f} AR={ar:.2f} F1={f1:.2f}")

            utils.log_to_csv(self.outdir / "test.csv",
                             total=self.iters, epoch=epoch, **gm)
            self.save_model(loss)

    def train_one_epoch(self):
        self.grid_metrics.reset()
        self.atom_metrics.reset()
        self.model.train()

        for i, (filenames, inps, targs, atoms) in enumerate(self.train_dtl):
            inps = inps.to(self.device, non_blocking=True)
            targs = targs.to(self.device, non_blocking=True)
            self.opt.zero_grad()

            preds = self.model(inps)
            loss, loss_values = self.model.compute_loss(preds, targs)
            loss.backward()

            grad = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.optimizer.clip_grad,
                error_if_nonfinite=False,
            )
            self.opt.step()

            fn = partial(
                box2atom,
                cell=self.cfg.dataset.real_size,
                threshold=0.5,
                cutoff=(1.8, 1.6, 1.5, 1.5),
                nms=self.cfg.dataset.nms,
                order=self.cfg.dataset.ion_type,
            )

            if self.cfg.setting.num_workers >= 1:
                with Pool(self.cfg.setting.num_workers) as p:
                    out_atoms = p.map(fn, preds.detach().cpu().numpy())
            else:
                out_atoms = list(map(fn, preds.detach().cpu().numpy()))

            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(
                loss=loss, grad=grad,
                conf=loss_values['conf'],
                xy=loss_values['xy'],
                z=loss_values['z'],
            )
            self.iters += 1

            if i % self.cfg.setting.log_every == 0:
                self.log.info(
                    f"  Train [{i:4d}/{len(self.train_dtl):4d}] "
                    f"L={loss.item():.4e} G={grad.item():.2e}"
                )

        self.scheduler.step()
        return self.grid_metrics.compute()

    @torch.no_grad()
    def test_one_epoch(self):
        self.grid_metrics.reset()
        self.atom_metrics.reset()
        self.model.eval()

        for i, (filenames, inps, targs, atoms) in enumerate(self.test_dtl):
            inps = inps.to(self.device, non_blocking=True)
            targs = targs.to(self.device, non_blocking=True)

            preds = self.model(inps)
            loss, loss_values = self.model.compute_loss(preds, targs)

            fn = partial(
                box2atom,
                cell=self.cfg.dataset.real_size,
                threshold=0.5,
                cutoff=(1.8, 1.6, 1.5, 1.5),
                nms=self.cfg.dataset.nms,
                order=self.cfg.dataset.ion_type,
            )
            out_atoms = list(map(fn, preds.detach().cpu().numpy()))
            self.atom_metrics.update(M=(out_atoms, atoms))
            self.grid_metrics.update(
                loss=loss,
                conf=loss_values['conf'],
                xy=loss_values['xy'],
                z=loss_values['z'],
            )

        return self.grid_metrics.compute(), self.atom_metrics.compute()

    def load_model(self):
        ckpt = self.cfg.model.checkpoint
        if ckpt and os.path.exists(ckpt):
            params = torch.load(ckpt, map_location=self.device)
            mismatch = self.model.load_state_dict(params, strict=False)
            self.log.info(f"加载模型: {ckpt}")
            if mismatch.missing_keys:
                self.log.info(f"  缺失键: {len(mismatch.missing_keys)}")
            if mismatch.unexpected_keys:
                self.log.info(f"  意外键: {len(mismatch.unexpected_keys)}")
        else:
            self.log.info("从头开始训练")

    def save_model(self, metric):
        if metric is None or metric < self.best:
            self.best = metric
            path = self.outdir / f"PROTEIN_E{self.epoch:03d}_L{metric:.3e}.pkl"
            if len(self.save_paths) >= self.cfg.setting.max_save:
                old = self.save_paths.pop(0)
                if os.path.exists(old):
                    os.remove(old)
            torch.save(self.model.state_dict(), path)
            self.save_paths.append(path)


def main():
    args = get_parser()
    cfg = Config()

    outdir = Path(args.outdir) / f"{time.strftime('%Y%m%d-%H%M%S')}-protein"
    cfg.setting.device = args.device
    cfg.setting.outdir = str(outdir)
    cfg.setting.debug = args.debug

    if cfg.setting.debug:
        cfg.setting.num_workers = 0
        cfg.setting.log_every = 1
        cfg.setting.batch_size = 2
        cfg.setting.max_save = 1
        cfg.setting.epoch = 3
        cfg.setting.pin_memory = False

    # ---- 平台自动配置 ----
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        n_threads = args.num_threads if args.num_threads > 0 else 4
        torch.set_num_threads(n_threads)

        if IS_WINDOWS:
            cfg.setting.num_workers = 0
            cfg.setting.pin_memory = False
        else:
            cfg.setting.num_workers = args.num_workers if args.num_workers >= 0 else 4
            cfg.setting.pin_memory = True

        print(f"GPU {args.gpu_id} | CPU线程: {n_threads} | "
              f"workers: {cfg.setting.num_workers} | batch_size={cfg.setting.batch_size}")

    elif args.device == "cpu":
        if IS_WINDOWS:
            # Windows: 不能用 DataLoader workers, 仅靠 OMP/MKL 线程并行
            n_threads = args.num_threads if args.num_threads > 0 else 8
            os.environ["OMP_NUM_THREADS"] = str(n_threads)
            os.environ["MKL_NUM_THREADS"] = str(n_threads)
            torch.set_num_threads(n_threads)
            cfg.setting.num_workers = 0
            cfg.setting.pin_memory = False
            print(f"[Windows] CPU 线程数: {torch.get_num_threads()} (OMP/MKL), "
                  f"batch_size={cfg.setting.batch_size}")
        else:
            # Linux CPU: DataLoader workers + OMP/MKL 线程双路并行
            n_threads = args.num_threads if args.num_threads > 0 else os.cpu_count() or 8
            os.environ["OMP_NUM_THREADS"] = str(n_threads)
            os.environ["MKL_NUM_THREADS"] = str(n_threads)
            torch.set_num_threads(n_threads)
            cfg.setting.num_workers = args.num_workers if args.num_workers >= 0 else 4
            cfg.setting.pin_memory = True
            print(f"[Linux] OMP/MKL 线程: {n_threads} | DataLoader workers: "
                  f"{cfg.setting.num_workers} | batch_size={cfg.setting.batch_size}")

    try:
        start = time.time()
        trainer = ProteinTrainer(cfg, train_ratio=args.train_ratio)
        trainer.fit()
        elapsed = (time.time() - start) / 3600
        print(f"\n训练完成! 总时间: {elapsed:.1f} 小时")
    except Exception as e:
        if not args.debug and time.time() - start < 300:
            shutil.rmtree(cfg.setting.outdir, ignore_errors=True)
        raise e


if __name__ == "__main__":
    main()
