"""
蛋白质 3D U-Net 检测配置
适配多元素蛋白质原子 (CA, C, N, O) 的 AFM-to-structure 重建
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

__all__ = ["ProteinDetectConfig"]


@dataclass
class SchedulerParams:
    step_size: int = 5
    gamma: float = 0.5


@dataclass
class Scheduler:
    name: str = "step"
    params: SchedulerParams = field(default_factory=SchedulerParams)


@dataclass
class ModelParams:
    # ---- 网络结构 ----
    in_size: Tuple[int, int, int] = (10, 100, 100)  # 10 通道, 100x100 像素
    in_channels: int = 1
    out_size: Tuple[int, int, int] = (8, 32, 32)    # Nz=8 (更深), 32x32 voxel
    # 3 元素(C,N,O) × 4 通道(conf+dx+dy+dz) = 12 输出通道
    out_channels: List[int] = field(default_factory=lambda: [12])
    model_channels: int = 32
    embedding_input: int = 0
    embedding_channels: int = 128
    num_res_blocks: Tuple[int, int] = (2, 2)
    attention_resolutions: List[int] = field(default_factory=lambda: [4, 8])
    dropout: float = 0.1
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    out_conv_blocks: int = 2
    out_mult: int = 1
    z_down: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    conv_resample: bool = True
    num_heads: int = 8
    activation: str = "silu"
    use_gated_conv: bool = False
    gated_conv_heads: Optional[int] = None

    # ---- 损失权重 ----
    cls_weight: float = 1.0
    xy_weight: float = 0.5
    z_weight: float = 0.5
    pos_weight: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])  # C, N, O


@dataclass
class Model:
    name: str = "UNetND-Protein"
    checkpoint: str = ""
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class TuneModelParams:
    """CycleGAN 调优模型 (可选, 用于噪声增强)"""
    in_size: Tuple[int, int, int] = (10, 100, 100)
    channels: int = 1
    out_conv_blocks: int = 1
    model_channels: int = 16
    num_res_blocks: List[int] = field(default_factory=lambda: [1, 1])
    attention_resolutions: List[int] = field(default_factory=lambda: [4, 8])
    dropout: float = 0.0
    gen_channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4])
    disc_channel_mult: List[int] = field(default_factory=lambda: [4, 8, 8])
    out_mult: int = 1
    gen_z_down: List[int] = field(default_factory=lambda: [2, 4, 8])
    disc_z_down: List[int] = field(default_factory=lambda: [])
    conv_resample: bool = True
    num_heads: int = 8
    activation: str = "silu"


@dataclass
class TuneModel:
    name: str = "CycleGAN"
    checkpoint: str = ""  # 空字符串 = 不使用
    params: TuneModelParams = field(default_factory=TuneModelParams)


@dataclass
class Setting:
    """训练设置 - 针对 14 核优化的参数"""
    epoch: int = 50
    batch_size: int = 4         # CPU 3D U-Net 内存受限
    num_workers: int = 0        # Windows 单进程 (OMP 线程替代)
    pin_memory: bool = False
    log_every: int = 50
    max_save: int = 5
    device: str = "cpu"           # 默认 CPU (14核)
    outdir: str = "outputs/"
    debug: bool = False


@dataclass
class Optimizer:
    lr: float = 1.0e-4
    weight_decay: float = 5.0e-4
    clip_grad: float = 5.0


@dataclass
class Dataset:
    """蛋白质数据集路径"""
    train_path: str = "dataset/protein_train"
    test_path: str = "dataset/protein_train"       # 同目录, 自动分割
    num_images: int = 10                # 10 通道 AFM 图像
    image_size: Tuple[int, int] = (100, 100)
    image_split: None = None           # 不使用分层采样
    # 蛋白质参数
    real_size: Tuple[float, float, float] = (25.0, 25.0, 8.0)
    box_size: Tuple[int, int, int] = (32, 32, 8)
    ion_type: List[str] = field(default_factory=lambda: ["C", "N", "O"])
    split: List[float] = field(default_factory=lambda: [0.0, 4.0, 8.0])
    nms: bool = True


@dataclass
class ProteinDetectConfig:
    model: Model = field(default_factory=Model)
    tune_model: TuneModel = field(default_factory=TuneModel)
    setting: Setting = field(default_factory=Setting)
    optimizer: Optimizer = field(default_factory=Optimizer)
    scheduler: Scheduler = field(default_factory=Scheduler)
    dataset: Dataset = field(default_factory=Dataset)
