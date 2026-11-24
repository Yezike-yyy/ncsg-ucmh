"""I2 前置诊断与 I1 逐位可靠记忆库在 UCCH 上的训练与评估主入口（baseline/ours/i1_train.py）。

设计原则与 M3-REVISE-001 / M3-REVISE-002 规范对齐：
  1. 支持变体：
     - off: 全关（纯 UCCH），不做额外前向、不消耗额外随机数、不做 epoch 末遍历，与 run_ucch.py 逐位完全一致；
     - probe: 先导探针（总视图数 K=4），训练使用原 MCR-A 单掩码可信度 pi_single 与标量记忆门控，
       epoch 末复用原第一视图随机流计算 c^1，并用独立探针流补充 c^2..c^4，仅输出诊断不改训练轨迹，且内置 c^1 一致性检查；
     - A-K1: 原 MCR-A 单掩码标量门控（K=1），在相同显式配置下与 mcr_train.py --mcr_a --mcr_a_warmup 1 逐位兼容；
     - A-K4: 多掩码均值可信度 pi_mean + 标量记忆门控（K=4）；
     - I1: 原始逐位门控，使用 pi_mean + 2K 跨模态位符号一致性 q[i, l]（K=4）；
     - RAND-Q: 原始等写入预算随机位对照（K=4），逐样本独立随机置换 q[i, :]；
     - SOFT-Q: 样本内均值归一化软位门控 g=(1-beta)+beta*q/mean(q)，零均值回退 g=1，固定 beta=0.5；
     - SOFT-RAND-Q: 先置换 q 再计算 g，保持每样本 g 分布和平均写入系数，隔离位身份作用；
     - A-K4-V / I1-V: 均值+方差可信度 pi_var 分支（通过 --variant A-K4-V / I1-V 或 --use_var 开启，默认关闭）；
     - L-pi: 仅 pi_single 损失加权，记忆保持 UCCH 标量动量；
     - M-pi: 仅 pi_single 记忆门控，Lc/Lr 使用原始无权重损失；
     - PERM-pi: 双门控下每轮独立置换训练 pi_single 快照，原始缓冲不覆盖。
  2. 预热与跨轮缓冲时序：
     - --mcr_a_warmup 默认固定为 1；预热轮（epoch < mcr_a_warmup）固定 pi=1, q=1；
     - 每轮训练和验证（test_epoch(epoch, is_eval=True)）完成后，在 eval + no_grad 下顺序遍历训练集一次，
       共享完整表示 (h_x, h_y) 与 K 个掩码视图计算下一轮使用的 pi_buf 与 q_buf（缓冲滞后一轮）。
  3. 随机流严格隔离：
     - 掩码随机流（mcr_mask_gen）、探针补充视图随机流（probe_mask_gen）、顺序遍历 DataLoader 随机流（mcr_loader_gen）、
       RAND-Q 位置换随机流（randq_gen）、GMM 随机种子、文本置换随机流均相互独立且不消耗全局 torch/numpy/random RNG。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

# 禁止在 upstream/ 目录内生成 __pycache__ 字节码文件，确保 git -C upstream status --short 始终为空
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.optim as optim  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]
from torch.nn.utils.clip_grad import clip_grad_norm  # type: ignore[import-not-found]

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent
UPSTREAM = BASELINE_DIR / "upstream"

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from ours.i1 import (  # type: ignore[import-not-found]
    DEFAULT_I2_LAMBDA_GRID,
    DEFAULT_SOFT_BETA,
    I1NCEAverage,
    I1Variant,
    VALID_I1_VARIANTS,
    apply_feature_mask,
    build_correspondence_permutation,
    compute_consistency_signal,
    compute_i2_reliabilities,
    compute_multiview_i2_signals,
    compute_per_sample_lr_signal,
    compute_sample_weights,
    compute_soft_bit_gate,
    inject_correspondence_noise,
    normalize_variant_name,
    permute_pi_single,
    permute_q_per_sample,
    prepare_applied_pi,
    resolve_dual_path_flags,
    summarize_i1_bitwise_diagnostics,
    summarize_i2_diagnostics,
    weighted_contrastive_loss,
    weighted_nce_softmax_loss,
)
from ours.separation import apply_separation_gate  # type: ignore[import-not-found]
from ours.label_diagnostics import compute_label_inconsistency  # type: ignore[import-not-found]
from run_ucch import (  # type: ignore[import-not-found]
    apply_seed,
    build_epoch_records,
    ensure_compat_modules,
    link_data_directory,
    make_map_wrapper,
    patch_torch_load_for_pt26,
    remove_data_link,
    resolve_data_root,
)

UPSTREAM_PARAM_KEYS: Tuple[str, ...] = (
    "data_name",
    "root_dir",
    "log_name",
    "pretrain",
    "pretrain_dir",
    "arch",
    "lr",
    "wd",
    "train_batch_size",
    "eval_batch_size",
    "max_epochs",
    "log_interval",
    "num_workers",
    "resume",
    "num_hiden_layers",
    "ls",
    "bit",
    "optimizer",
    "alpha",
    "momentum",
    "K",
    "T",
    "shift",
    "margin",
    "warmup_epoch",
)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建包含上游全部参数 + 包装器参数 + I2/I1 与旧 MCR 兼容参数的命令行解析器。"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- 1. 上游 utils/config.py 全部参数（同名、同默认值） ----
    parser.add_argument("--data_name", type=str, default="iapr_fea", help="data name")
    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--log_name", type=str, default="UCCH")
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_dir", type=str, default="UCCH")
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="vgg11",
        help="model architecture: ResNet | VGG (default: vgg11)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--num_hiden_layers",
        default=[3, 2],
        nargs="+",
        help="<Required> Number of hiden lyaers",
    )
    parser.add_argument("--ls", type=str, default="linear", help="lr scheduler")
    parser.add_argument("--bit", type=int, default=32, help="output shape")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.4)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--T", type=float, default=0.9)
    parser.add_argument("--shift", type=float, default=1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--warmup_epoch", type=int, default=1)

    # ---- 2. 包装控制参数 ----
    parser.add_argument("--seed", type=int, default=123, help="全局随机种子（默认 123，与上游一致）")
    parser.add_argument("--out", type=str, required=True, help="实验输出目录（存放 summary.json、诊断与缓冲快照）")
    parser.add_argument("--data_root", type=str, default="", help="数据集根目录")
    parser.add_argument("--deterministic", action="store_true", help="开启确定性模式")
    parser.add_argument("--no_save_ckpt", action="store_true", help="不把最佳 checkpoint 写入磁盘以节省空间")
    parser.add_argument(
        "--dump_codes",
        nargs="?",
        const="test_codes.npz",
        default="",
        help="将最终测试集的 query/retrieval 二值哈希码保存为 npz 文件（默认保存到 <out>/test_codes.npz）",
    )

    # ---- 3. I2 / I1 变体与超参数（及 MCR-A 兼容参数） ----
    parser.add_argument(
        "--variant",
        "--i1_variant",
        dest="variant",
        type=str,
        default="off",
        choices=[*VALID_I1_VARIANTS, "A"],
        help="运行变体：off | probe | A-K1 | A-K4 | I1 | RAND-Q | SOFT-Q | SOFT-RAND-Q | A-K4-V | I1-V | L-pi | M-pi | PERM-pi（默认 off）",
    )
    parser.add_argument(
        "--mcr_a",
        action="store_true",
        default=False,
        help="兼容旧 MCR-A 开关：当 --variant 为 off 时自动映射为 A-K1",
    )
    parser.add_argument(
        "--use_var",
        "--i2_use_var",
        dest="use_var",
        action="store_true",
        default=False,
        help="开启 I2 均值+方差可信度分支（默认关闭；开启时 A-K4->A-K4-V, I1->I1-V, RAND-Q 使用 pi_var）",
    )
    parser.add_argument(
        "--num_views",
        "--i1_k",
        dest="num_views",
        type=int,
        default=4,
        help="多视图掩码次数 K（多视图变体默认 4；A-K1 固定为 1）",
    )
    parser.add_argument(
        "--soft_beta",
        type=float,
        default=DEFAULT_SOFT_BETA,
        help="Soft-Q 样本内均值归一化软位门控强度（预声明固定 0.5；0 退化 A-K4）",
    )
    parser.add_argument(
        "--i2_lambda",
        type=float,
        default=1.0,
        help="I2 均值+方差分支的阻尼系数 lambda（工作初值默认 1.0）",
    )
    parser.add_argument(
        "--i2_lambda_grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_I2_LAMBDA_GRID),
        help="I2 等预算敏感性诊断候选 lambda 列表（默认 0.0 0.5 1.0 2.0）",
    )
    parser.add_argument(
        "--mcr_signal",
        type=str,
        choices=["consistency", "loss"],
        default="consistency",
        help="可信度估计信号类型（默认 consistency）",
    )
    parser.add_argument(
        "--mcr_a_warmup",
        type=int,
        default=1,
        help="预热 epoch 数（M3 固定默认 1，预热轮 pi=q=1）",
    )
    parser.add_argument("--mcr_gamma", type=float, default=0.0, help="样本权重难例调制指数 gamma（默认 0.0）")
    parser.add_argument("--mcr_delta", type=float, default=0.1, help="噪声分量残余地板权重 delta（默认 0.1）")
    parser.add_argument(
        "--mcr_normalize",
        type=str,
        choices=["n", "sum_w"],
        default="n",
        help="加权损失归一化方式：'n' 或 'sum_w'（默认 'n'）",
    )
    parser.add_argument("--mcr_rho_bar", type=float, default=0.8, help="固定特征掩码保留率 rho_bar（默认 0.8）")
    parser.add_argument("--mcr_mask_rescale", action="store_true", default=False, help="掩码后是否除以 rho 做期望缩放（默认 False）")
    parser.add_argument("--mcr_seed_offset", type=int, default=1000, help="专用随机数种子相对 --seed 的偏移量（默认 1000）")
    parser.add_argument("--probe_seed_offset", type=int, default=100, help="probe 补充视图 c2..cK 相对 mcr_seed 的偏移量（默认 100）")
    parser.add_argument("--randq_seed_offset", type=int, default=200, help="RAND-Q 位置换流相对 mcr_seed 的偏移量（默认 200）")
    parser.add_argument("--perm_seed_offset", type=int, default=300, help="PERM-pi 训练 pi 缓冲置换流相对 mcr_seed 的偏移量（默认 300）")
    parser.add_argument("--noise_rate", type=float, default=0.0, help="训练集文本置换率 eta，不等于实际标签不一致率（默认 0.0）")
    parser.add_argument("--noise_seed", type=int, default=0, help="跨模态对应噪声置换的独立随机种子（默认 0）")

    parser.add_argument("--sep_gate_mode", choices=["ashman", "weight"], default="ashman")
    parser.add_argument("--sep_d_lo", type=float, default=1.0)
    parser.add_argument("--sep_d_hi", type=float, default=2.0)
    parser.add_argument("--sep_weight_thr", type=float, default=0.85)
    return parser


def parse_all_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    """支持带 `--` 分隔符或不带 `--` 的全部参数解析，并提取透传给上游的参数列表。"""
    cleaned_argv = [tok for tok in argv if tok != "--"]
    # argparse 默认不识别带负指数形式的独立负数参数。
    for name in ("--sep_d_lo", "--sep_d_hi"):
        if name in cleaned_argv:
            index = cleaned_argv.index(name)
            if index + 1 < len(cleaned_argv) and cleaned_argv[index + 1].startswith("-"):
                try:
                    float(cleaned_argv[index + 1])
                except ValueError:
                    pass
                else:
                    cleaned_argv[index:index + 2] = [name + "=" + cleaned_argv[index + 1]]
    parser = build_arg_parser()
    args = parser.parse_args(cleaned_argv)
    args.num_hiden_layers = [int(i) for i in args.num_hiden_layers]
    args.i2_lambda_grid = [float(x) for x in args.i2_lambda_grid]
    args.variant = normalize_variant_name(
        args.variant,
        use_var=bool(args.use_var),
        mcr_a_flag=bool(args.mcr_a),
    )
    if args.variant in ("A-K4-V", "I1-V"):
        args.use_var = True
    if args.variant in ("SOFT-Q", "SOFT-RAND-Q"):
        if args.use_var:
            raise ValueError("Soft-Q 与 SOFT-RAND-Q 固定使用 pi_mean，不支持 --use_var")
        if not 0.0 <= float(args.soft_beta) <= 1.0:
            raise ValueError(f"soft_beta 必须位于 [0, 1]，当前为 {args.soft_beta}")

    if "--" in argv:
        cut = list(argv).index("--")
        upstream_passthrough = list(argv[cut + 1 :])
    else:
        upstream_passthrough = list(cleaned_argv)

    return args, upstream_passthrough


def resolve_effective_k_views(variant: I1Variant, configured_k: int) -> int:
    """根据变体解析实际掩码视图数 K。"""
    if variant == "off":
        return 0
    if variant in ("A-K1", "A-K1-G", "L-pi", "M-pi", "PERM-pi"):
        return 1
    if configured_k < 1:
        raise ValueError(f"num_views (K) 必须 >= 1，当前为 {configured_k}")
    if variant == "probe" and configured_k < 2:
        raise ValueError(f"probe 变体要求 K >= 2（标准为 K=4），当前为 {configured_k}")
    return int(configured_k)


def resolve_pi_source_name(variant: I1Variant, use_var: bool) -> str:
    """返回当前变体用于训练样本权重与记忆库门控的 pi 来源名称。"""
    if variant == "off":
        return "none"
    if variant in ("probe", "A-K1", "A-K1-G", "L-pi", "M-pi", "PERM-pi"):
        return "pi_single"
    if variant in ("A-K4-V", "I1-V") or (variant == "RAND-Q" and use_var):
        return "pi_var"
    return "pi_mean"


def build_epoch_diag_entry(
    epoch: int,
    variant: I1Variant,
    pi_source: str,
    train_loss_mean: float,
    lc_loss_mean: float,
    lr_loss_mean: float,
    epoch_seconds: float,
    a_warmup_active: bool,
    applied_snapshot_from_epoch: Optional[int],
    applied_pi_arr: np.ndarray,
    applied_q_arr: np.ndarray,
    applied_write_budget_diag: Dict[str, Any],
    next_pi_arr: np.ndarray,
    gmm_clean_mean: Optional[float],
    gmm_noisy_mean: Optional[float],
    noise_mask: np.ndarray,
    label_inconsistent_mask: np.ndarray,
    i2_diag: Optional[Dict[str, Any]],
    i1_diag: Optional[Dict[str, Any]],
    perm_diag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """整理单个 epoch 的兼容 MCR 诊断字段 + I2/I1 新增诊断字段。"""
    q10, q25, q50, q75, q90 = [
        float(v) for v in np.quantile(next_pi_arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    ]

    def safe_auc(target_mask: np.ndarray) -> Optional[float]:
        if not 0 < int(np.sum(target_mask)) < int(target_mask.shape[0]):
            return None
        try:
            return float(roc_auc_score((~target_mask).astype(np.int32), next_pi_arr))
        except ValueError:
            return 0.5

    clean_sel = ~noise_mask
    label_consistent_sel = noise_mask & ~label_inconsistent_mask
    pi_clean_mean = float(np.mean(next_pi_arr[clean_sel])) if np.any(clean_sel) else None
    pi_noisy_mean = float(np.mean(next_pi_arr[noise_mask])) if np.any(noise_mask) else None
    pi_label_inconsistent_mean = (
        float(np.mean(next_pi_arr[label_inconsistent_mask]))
        if np.any(label_inconsistent_mask)
        else None
    )
    pi_label_consistent_mean = (
        float(np.mean(next_pi_arr[label_consistent_sel]))
        if np.any(label_consistent_sel)
        else None
    )

    return {
        "epoch": int(epoch),
        "variant": str(variant),
        "pi_source": str(pi_source),
        "train_loss_mean": float(train_loss_mean),
        "lc_loss_mean": float(lc_loss_mean),
        "lr_loss_mean": float(lr_loss_mean),
        "lm_loss_mean": 0.0,
        "epoch_seconds": round(float(epoch_seconds), 4),
        "a_warmup_active": bool(a_warmup_active),
        "applied_snapshot_from_epoch": applied_snapshot_from_epoch,
        "applied_pi_mean": float(np.mean(applied_pi_arr)),
        "applied_q_mean": float(np.mean(applied_q_arr)),
        "applied_write_budget_1_minus_m": applied_write_budget_diag,
        "pi_mean": float(np.mean(next_pi_arr)),
        "pi_std": float(np.std(next_pi_arr)),
        "pi_quantiles": {
            "q10": q10,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q90": q90,
        },
        "gmm_clean_mean": gmm_clean_mean,
        "gmm_noisy_mean": gmm_noisy_mean,
        "gmm_means": (
            [gmm_clean_mean, gmm_noisy_mean]
            if (gmm_clean_mean is not None and gmm_noisy_mean is not None)
            else None
        ),
        "pi_auc": safe_auc(noise_mask),
        "pi_label_inconsistent_auc": safe_auc(label_inconsistent_mask),
        "pi_clean_mean": pi_clean_mean,
        "pi_noisy_mean": pi_noisy_mean,
        "pi_permuted_mean": pi_noisy_mean,
        "pi_label_inconsistent_mean": pi_label_inconsistent_mean,
        "pi_label_consistent_mean": pi_label_consistent_mean,
        "label_inconsistent_fraction": float(np.mean(label_inconsistent_mask)),
        "perm_pi": perm_diag if perm_diag is not None else {"enabled": False},
        "i2": i2_diag,
        "i1": i1_diag,
    }


def run_training(
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    """严格按上游 UCCH.main() 的顺序执行数据集构建、模型初始化、训练、验证测试与 epoch 末多视图遍历。"""
    models = importlib.import_module("nets")
    utils = importlib.import_module("src.utils")
    nce_avg_mod = importlib.import_module("NCE.NCEAverage")
    nce_crit_mod = importlib.import_module("NCE.NCECriterion")
    cmdataset_mod = importlib.import_module("src.cmdataset")
    tb_mod = importlib.import_module("torch.utils.tensorboard")

    NCEAverage: Any = getattr(nce_avg_mod, "NCEAverage")
    NCESoftmaxLoss: Any = getattr(nce_crit_mod, "NCESoftmaxLoss")
    CMDataset: Any = getattr(cmdataset_mod, "CMDataset")
    SummaryWriter: Any = getattr(tb_mod, "SummaryWriter")

    args.root_dir = str(out_dir)
    args.log_dir = os.path.join(args.root_dir, "logs", args.log_name)
    args.ckpt_dir = os.path.join(args.root_dir, "ckpt", args.pretrain_dir)
    try:
        os.makedirs(args.log_dir, exist_ok=True)
        if not args.no_save_ckpt:
            os.makedirs(args.ckpt_dir, exist_ok=True)
    except OSError as exc:
        sys.exit(f"创建日志或检查点目录失败: {exc}")

    print("===> Preparing data ..")
    train_dataset = CMDataset(
        args.data_name,
        return_index=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    retrieval_dataset = CMDataset(
        args.data_name,
        partition="retrieval",
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = CMDataset(
        args.data_name,
        partition="test",
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("===> Building ResNet..")
    image_net_cls: Any = getattr(models, "ImageNet")
    text_net_cls: Any = getattr(models, "TextNet")
    if "fea" in args.data_name:
        image_model: nn.Module = image_net_cls(
            y_dim=train_dataset.imgs.shape[1],
            bit=args.bit,
            hiden_layer=args.num_hiden_layers[0],
        ).cuda()
        backbone: Optional[nn.Module] = None
    else:
        arch_cls: Any = getattr(models, args.arch)
        backbone_mod: nn.Module = arch_cls(pretrained=args.pretrain, feature=True).cuda()
        backbone = backbone_mod
        fea_dim = (
            4096
            if "vgg" in args.arch.lower()
            else (512 if args.arch == "resnet18" or args.arch == "resnet34" else 2048)
        )
        fea_net: nn.Module = image_net_cls(
            y_dim=fea_dim,
            bit=args.bit,
            hiden_layer=args.num_hiden_layers[0],
        ).cuda()
        image_model = nn.Sequential(backbone_mod, fea_net)

    text_model: nn.Module = text_net_cls(
        y_dim=train_dataset.text_dim,
        bit=args.bit,
        hiden_layer=args.num_hiden_layers[1],
    ).cuda()

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    wd = args.wd
    if args.optimizer == "SGD":
        optimizer: optim.Optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)

    if args.ls == "cos":
        lr_schedu: Any = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epochs, eta_min=0, last_epoch=-1
        )
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.1)

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        image_model.load_state_dict(ckpt["image_model_state_dict"])
        text_model.load_state_dict(ckpt["text_model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"])
        print("===> Load last checkpoint data")
    else:
        start_epoch = 0
        print("===> Start from scratch")

    def set_train(is_warmup: bool = False) -> None:
        image_model.train()
        if is_warmup and backbone is not None:
            backbone.eval()
            backbone.requires_grad_(False)
        elif backbone is not None:
            backbone.requires_grad_(True)
        text_model.train()

    def set_eval() -> None:
        image_model.eval()
        text_model.eval()

    criterion = utils.ContrastiveLoss(args.margin, shift=args.shift)
    n_data = len(train_dataset)
    contrast = NCEAverage(args.bit, n_data, args.K, args.T, args.momentum)
    criterion_contrast = NCESoftmaxLoss()
    contrast = contrast.cuda()
    criterion_contrast = criterion_contrast.cuda()

    # ---- 在所有上游模型与记忆库初始化完毕后，执行独立种子的噪声注入与 I2/I1 缓冲区初始化 ----
    noise_mask = inject_correspondence_noise(
        train_dataset,
        noise_rate=float(args.noise_rate),
        noise_seed=int(args.noise_seed),
    )
    _, source_indices = build_correspondence_permutation(
        n_data,
        float(args.noise_rate),
        int(args.noise_seed),
    )
    label_inconsistent_mask, label_diagnostic = compute_label_inconsistency(
        train_dataset.labels,
        noise_mask,
        source_indices,
    )
    label_diagnostic["text_permutation_rate_eta"] = float(args.noise_rate)
    label_diagnostic["noise_seed"] = int(args.noise_seed)
    try:
        with open(out_dir / "noise_mask.npy", "wb") as f_nm:
            np.save(f_nm, noise_mask)
        with open(out_dir / "noise_sources.npy", "wb") as f_ns:
            np.save(f_ns, source_indices)
        with open(out_dir / "label_inconsistency.json", "w", encoding="utf-8") as f_diag:
            json.dump(label_diagnostic, f_diag, indent=2, ensure_ascii=False)
    except OSError as exc:
        sys.exit(f"保存文本置换与标签诊断失败 ({out_dir}): {exc}")

    variant: I1Variant = args.variant
    i1_active = bool(variant != "off")
    bitwise_active = bool(variant in ("I1", "RAND-Q", "SOFT-Q", "SOFT-RAND-Q", "I1-V"))
    is_rand_q = bool(variant in ("RAND-Q", "SOFT-RAND-Q"))
    soft_active = bool(variant in ("SOFT-Q", "SOFT-RAND-Q"))
    is_probe = bool(variant == "probe")
    loss_weighting, memory_gating, perm_pi_enabled = resolve_dual_path_flags(variant)
    k_views = resolve_effective_k_views(variant, int(args.num_views))
    pi_source_name = resolve_pi_source_name(variant, bool(args.use_var))

    norm_mode: Literal["n", "sum_w"] = "sum_w" if args.mcr_normalize == "sum_w" else "n"
    signal_mode: Literal["consistency", "loss"] = "loss" if args.mcr_signal == "loss" else "consistency"

    mcr_seed = int(args.seed) + int(args.mcr_seed_offset)
    probe_seed = mcr_seed + int(args.probe_seed_offset)
    randq_seed = mcr_seed + int(args.randq_seed_offset)
    perm_seed = mcr_seed + int(args.perm_seed_offset)
    mcr_device = "cuda" if torch.cuda.is_available() else "cpu"

    mcr_mask_gen = torch.Generator(device=mcr_device)
    mcr_mask_gen.manual_seed(mcr_seed)
    probe_mask_gen = torch.Generator(device=mcr_device)
    probe_mask_gen.manual_seed(probe_seed)
    randq_gen = torch.Generator(device=mcr_device)
    randq_gen.manual_seed(randq_seed)
    perm_gen = torch.Generator(device=mcr_device)
    perm_gen.manual_seed(perm_seed)
    mcr_loader_gen = torch.Generator(device="cpu")
    mcr_loader_gen.manual_seed(mcr_seed + 1)

    eval_train_loader: Optional[torch.utils.data.DataLoader[Any]] = None
    if i1_active:
        eval_workers = 0 if "fea" in args.data_name else int(args.num_workers)
        eval_train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.eval_batch_size,
            num_workers=eval_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            generator=mcr_loader_gen,
        )

    # 首个 epoch 开始前不生成门控，预热轮固定 pi=1, q=1；缓冲滞后一轮
    pi_buf = torch.ones(n_data, dtype=torch.float32, device=mcr_device)
    applied_pi_buf = pi_buf
    q_buf = torch.ones((n_data, int(args.bit)), dtype=torch.float32, device=mcr_device)
    q_raw_last_np = np.ones((n_data, int(args.bit)), dtype=np.float32)
    snapshot_source_epoch: Optional[int] = None

    i1_nce = I1NCEAverage(contrast)

    records: List[Dict[str, Any]] = []
    map_calc = make_map_wrapper(records)
    best_acc = 0.0
    best_epoch = start_epoch
    epoch_diagnostics: List[Dict[str, Any]] = []

    def train_one_epoch(epoch: int) -> Tuple[float, float, float]:
        print("\nEpoch: %d / %d" % (epoch, args.max_epochs))
        set_train(epoch < args.warmup_epoch)
        i1_nce.reset_epoch_stats()
        train_loss = 0.0
        lc_sum = 0.0
        lr_sum = 0.0
        n_batches = 0

        # 全关退化路径：严格走上游原始类与原始计算，不做任何额外张量操作或随机数调用
        if not i1_active:
            for batch_idx, (idx, images, texts, _) in enumerate(train_loader):
                images, texts, idx = (
                    [img.cuda() for img in images],
                    [txt.cuda() for txt in texts],
                    [idx.cuda()],
                )
                images_outputs = [image_model(im) for im in images]
                texts_outputs = [text_model(txt.float()) for txt in texts]

                out_l, out_ab = contrast(
                    torch.cat(images_outputs),
                    torch.cat(texts_outputs),
                    torch.cat(idx * len(images)),
                    epoch=epoch - args.warmup_epoch,
                )
                l_loss = criterion_contrast(out_l)
                ab_loss = criterion_contrast(out_ab)
                Lc = l_loss + ab_loss
                Lr = criterion(torch.cat(images_outputs), torch.cat(texts_outputs))
                loss = Lc * args.alpha + Lr * (1.0 - args.alpha)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(parameters, 1.0)
                optimizer.step()

                train_loss += float(loss.item())
                lc_sum += float(Lc.item())
                lr_sum += float(Lr.item())
                n_batches += 1

                if batch_idx % args.log_interval == 0:
                    summary_writer.add_scalar(
                        "Loss/train",
                        train_loss / (batch_idx + 1),
                        epoch * len(train_loader) + batch_idx,
                    )
                    summary_writer.add_scalar(
                        "learning rate",
                        optimizer.param_groups[0]["lr"],
                        epoch * len(train_loader) + batch_idx,
                    )

            denom = max(n_batches, 1)
            return train_loss / denom, lc_sum / denom, lr_sum / denom

        # 激活路径（既有变体及 L-pi / M-pi / PERM-pi）
        a_warmup_now = bool(epoch < int(args.mcr_a_warmup))
        for batch_idx, (idx, images, texts, _) in enumerate(train_loader):
            images_gpu = [img.cuda() for img in images]
            texts_gpu = [txt.cuda() for txt in texts]
            idx_list = [idx.cuda()]

            images_outputs = [image_model(im) for im in images_gpu]
            texts_outputs = [text_model(txt.float()) for txt in texts_gpu]
            im_cat = torch.cat(images_outputs)
            txt_cat = torch.cat(texts_outputs)
            idx_cat = torch.cat(idx_list * len(images_gpu))
            idx_flat = idx_cat.view(-1)

            if a_warmup_now:
                batch_pi = torch.ones(idx_flat.size(0), dtype=im_cat.dtype, device=im_cat.device)
                batch_q: Optional[torch.Tensor] = None
            else:
                batch_pi = applied_pi_buf.index_select(0, idx_flat).to(dtype=im_cat.dtype)
                if bitwise_active:
                    batch_q = q_buf.index_select(0, idx_flat).to(dtype=im_cat.dtype)
                else:
                    batch_q = None

            w_batch = (
                compute_sample_weights(
                    batch_pi,
                    im_cat,
                    txt_cat,
                    gamma=float(args.mcr_gamma),
                    delta=float(args.mcr_delta),
                )
                if loss_weighting
                else None
            )
            nce_pi = batch_pi if memory_gating else None
            nce_q = batch_q if memory_gating else None
            out_l, out_ab = i1_nce(
                im_cat,
                txt_cat,
                idx_cat,
                epoch=epoch - args.warmup_epoch,
                pi=nce_pi,
                q=nce_q,
                track_scalar_direction=soft_active,
            )
            if loss_weighting:
                assert w_batch is not None
                l_loss = weighted_nce_softmax_loss(out_l, w_batch, normalize=norm_mode)
                ab_loss = weighted_nce_softmax_loss(out_ab, w_batch, normalize=norm_mode)
                Lc = l_loss + ab_loss
                Lr = weighted_contrastive_loss(
                    im_cat,
                    txt_cat,
                    w_batch,
                    margin=float(args.margin),
                    shift=float(args.shift),
                    normalize=norm_mode,
                )
            else:
                l_loss = criterion_contrast(out_l)
                ab_loss = criterion_contrast(out_ab)
                Lc = l_loss + ab_loss
                Lr = criterion(im_cat, txt_cat)
            loss = Lc * args.alpha + Lr * (1.0 - args.alpha)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters, 1.0)
            optimizer.step()

            train_loss += float(loss.item())
            lc_sum += float(Lc.item())
            lr_sum += float(Lr.item())
            n_batches += 1

            if batch_idx % args.log_interval == 0:
                summary_writer.add_scalar(
                    "Loss/train",
                    train_loss / (batch_idx + 1),
                    epoch * len(train_loader) + batch_idx,
                )
                summary_writer.add_scalar(
                    "learning rate",
                    optimizer.param_groups[0]["lr"],
                    epoch * len(train_loader) + batch_idx,
                )

        denom = max(n_batches, 1)
        return train_loss / denom, lc_sum / denom, lr_sum / denom

    def eval_codes(data_loader: torch.utils.data.DataLoader[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        imgs: List[torch.Tensor] = []
        txts: List[torch.Tensor] = []
        labs: List[torch.Tensor] = []
        with torch.no_grad():
            for _, (images, texts, targets) in enumerate(data_loader):
                images_gpu = [img.cuda() for img in images]
                texts_gpu = [txt.cuda() for txt in texts]
                targets_gpu = targets.cuda()

                images_outputs = [image_model(im) for im in images_gpu]
                texts_outputs = [text_model(txt.float()) for txt in texts_gpu]

                imgs += images_outputs
                txts += texts_outputs
                labs.append(targets_gpu)

            imgs_np = torch.cat(imgs).sign_().cpu().numpy()
            txts_np = torch.cat(txts).sign_().cpu().numpy()
            labs_np = torch.cat(labs).cpu().numpy()
        return imgs_np, txts_np, labs_np

    def test_epoch(epoch: int, is_eval: bool = True) -> Tuple[float, float]:
        nonlocal best_acc, best_epoch
        set_eval()
        retrieval_imgs, retrieval_txts, retrieval_labs = eval_codes(retrieval_loader)
        if is_eval:
            query_imgs, query_txts, query_labs = (
                retrieval_imgs[0:2000],
                retrieval_txts[0:2000],
                retrieval_labs[0:2000],
            )
            retrieval_imgs, retrieval_txts, retrieval_labs = (
                retrieval_imgs[0:2000],
                retrieval_txts[0:2000],
                retrieval_labs[0:2000],
            )
        else:
            query_imgs, query_txts, query_labs = eval_codes(query_loader)
            if args.dump_codes:
                dump_target = Path(args.dump_codes)
                if not dump_target.is_absolute():
                    dump_target = out_dir / dump_target
                try:
                    dump_target.parent.mkdir(parents=True, exist_ok=True)
                    with open(dump_target, "wb") as f_npz:
                        np.savez(
                            f_npz,
                            retrieval_imgs=retrieval_imgs,
                            retrieval_txts=retrieval_txts,
                            retrieval_labs=retrieval_labs,
                            query_imgs=query_imgs,
                            query_txts=query_txts,
                            query_labs=query_labs,
                        )
                except OSError as exc:
                    print(f"[警告] 保存测试集二值码失败 ({dump_target}): {exc}", file=sys.stderr)

        i2t = map_calc(retrieval_txts, retrieval_labs, query_imgs, query_labs, 0, "hamming")
        t2i = map_calc(retrieval_imgs, retrieval_labs, query_txts, query_labs, 0, "hamming")

        avg = (i2t + t2i) / 2.0
        print(
            "%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g"
            % ("Evaluation" if is_eval else "Test", i2t, t2i, avg)
        )
        if avg > best_acc:
            print("Saving..")
            state = {
                "image_model_state_dict": image_model.state_dict(),
                "text_model_state_dict": text_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "Avg": avg,
                "Img2Txt": i2t,
                "Txt2Img": t2i,
                "epoch": epoch,
            }
            if not args.no_save_ckpt:
                ckpt_file = os.path.join(args.ckpt_dir, "%s_%d_best_checkpoint.t7" % (args.arch, args.bit))
                try:
                    torch.save(state, ckpt_file)
                except OSError as exc:
                    print(f"[警告] 保存检查点失败 ({ckpt_file}): {exc}", file=sys.stderr)
            best_acc = avg
            best_epoch = epoch
        return i2t, t2i

    def run_epoch_end_i1_traversal(
        epoch: int,
        mem_signs_before: np.ndarray,
        mem_signs_after: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        """每个 epoch 验证结束后，在 eval + no_grad 下顺序遍历训练集一次，
        共享完整表示 (h_x, h_y) 与 K 个掩码视图，同时计算 I2 信号与 I1 位一致性 q，
        并更新供下一轮（epoch + 1）使用的 pi_buf 与 q_buf。
        """
        nonlocal q_raw_last_np, snapshot_source_epoch
        assert eval_train_loader is not None
        set_eval()

        all_c_views = torch.zeros((n_data, k_views), dtype=torch.float32, device=mcr_device)
        all_c_bar = torch.zeros(n_data, dtype=torch.float32, device=mcr_device)
        all_s_std = torch.zeros(n_data, dtype=torch.float32, device=mcr_device)
        all_q_raw = torch.ones((n_data, int(args.bit)), dtype=torch.float32, device=mcr_device)

        # 在 probe 或 A-K1 模式下，克隆 mcr_mask_gen 初始状态以独立核验第一视图 c^1 与单掩码流逐位一致
        c1_check_gen: Any = None
        all_c1_ref: Any = None
        if is_probe or variant == "A-K1":
            _gen = torch.Generator(device=mcr_device)
            _gen.set_state(mcr_mask_gen.get_state().clone())
            c1_check_gen = _gen
            all_c1_ref = torch.zeros(n_data, dtype=torch.float32, device=mcr_device)

        with torch.no_grad():
            for _, (idx, images, texts, _) in enumerate(eval_train_loader):
                idx_gpu = idx.cuda().view(-1)
                images_gpu = [img.cuda() for img in images]
                texts_gpu = [txt.cuda() for txt in texts]

                # 完整表示只计算一次
                h_x = torch.cat([image_model(im) for im in images_gpu])
                h_y = torch.cat([text_model(txt.float()) for txt in texts_gpu])

                # 若开启 c^1 核验，用克隆的独立单掩码流计算参考 c^1
                if c1_check_gen is not None and all_c1_ref is not None:
                    hx_ref = torch.cat(
                        [
                            image_model(
                                apply_feature_mask(
                                    im.float(),
                                    float(args.mcr_rho_bar),
                                    c1_check_gen,
                                    rescale=bool(args.mcr_mask_rescale),
                                )
                            )
                            for im in images_gpu
                        ]
                    )
                    hy_ref = torch.cat(
                        [
                            text_model(
                                apply_feature_mask(
                                    txt.float(),
                                    float(args.mcr_rho_bar),
                                    c1_check_gen,
                                    rescale=bool(args.mcr_mask_rescale),
                                )
                            )
                            for txt in texts_gpu
                        ]
                    )
                    if signal_mode == "loss":
                        c1_ref_batch = compute_per_sample_lr_signal(
                            h_x,
                            h_y,
                            margin=float(args.margin),
                            shift=float(args.shift),
                        )
                    else:
                        c1_ref_batch = compute_consistency_signal(h_x, h_y, hx_ref, hy_ref)
                    all_c1_ref.index_copy_(0, idx_gpu, c1_ref_batch.to(dtype=all_c1_ref.dtype))

                # 共享的 K 次掩码视图：
                # - probe 模式：第 1 视图 (k=0) 使用 mcr_mask_gen，第 2..K 视图 (k>=1) 使用独立 probe_mask_gen；
                # - 其余模式：全部 k=0..K-1 视图均使用 mcr_mask_gen。
                hx_views_batch: List[torch.Tensor] = []
                hy_views_batch: List[torch.Tensor] = []
                for k_idx in range(k_views):
                    active_gen = (
                        mcr_mask_gen
                        if (not is_probe or k_idx == 0)
                        else probe_mask_gen
                    )
                    hx_k = torch.cat(
                        [
                            image_model(
                                apply_feature_mask(
                                    im.float(),
                                    float(args.mcr_rho_bar),
                                    active_gen,
                                    rescale=bool(args.mcr_mask_rescale),
                                )
                            )
                            for im in images_gpu
                        ]
                    )
                    hy_k = torch.cat(
                        [
                            text_model(
                                apply_feature_mask(
                                    txt.float(),
                                    float(args.mcr_rho_bar),
                                    active_gen,
                                    rescale=bool(args.mcr_mask_rescale),
                                )
                            )
                            for txt in texts_gpu
                        ]
                    )
                    hx_views_batch.append(hx_k)
                    hy_views_batch.append(hy_k)

                c_views_b, c_bar_b, s_std_b, q_b = compute_multiview_i2_signals(
                    h_x,
                    h_y,
                    hx_views_batch,
                    hy_views_batch,
                )
                if signal_mode == "loss":
                    lr_sig = compute_per_sample_lr_signal(
                        h_x,
                        h_y,
                        margin=float(args.margin),
                        shift=float(args.shift),
                    ).to(dtype=torch.float32)
                    c_views_b = lr_sig.unsqueeze(1).expand(lr_sig.size(0), k_views).contiguous()
                    c_bar_b = lr_sig
                    s_std_b = torch.zeros_like(lr_sig)

                all_c_views.index_copy_(0, idx_gpu, c_views_b)
                all_c_bar.index_copy_(0, idx_gpu, c_bar_b)
                all_s_std.index_copy_(0, idx_gpu, s_std_b)
                all_q_raw.index_copy_(0, idx_gpu, q_b)

        probe_c1_check: Optional[Dict[str, Any]] = None
        if c1_check_gen is not None and all_c1_ref is not None:
            c1_actual = all_c_views[:, 0]
            max_diff = float((c1_actual - all_c1_ref).abs().max().item())
            rng_state_eq = bool(torch.equal(mcr_mask_gen.get_state(), c1_check_gen.get_state()))
            c1_eq = bool(torch.equal(c1_actual, all_c1_ref)) and rng_state_eq
            probe_c1_check = {
                "checked": True,
                "matches_single_stream": c1_eq,
                "c1_tensor_bitwise_equal": bool(torch.equal(c1_actual, all_c1_ref)),
                "mcr_mask_gen_state_bitwise_equal": rng_state_eq,
                "max_abs_diff": max_diff,
            }
            if not c1_eq:
                raise RuntimeError(
                    f"probe/A-K1 第一视图 c^1 一致性检查失败: max_abs_diff={max_diff}, rng_state_eq={rng_state_eq}"
                )

        i2_res = compute_i2_reliabilities(
            all_c_views,
            s_std=all_s_std,
            lambda_var=float(args.i2_lambda),
            mode=signal_mode,
            random_state=mcr_seed,
            lambda_grid=args.i2_lambda_grid,
            include_gmm_params=variant == "A-K1-G",
        )

        if pi_source_name == "pi_single":
            next_pi_np = np.asarray(i2_res["pi_single"], dtype=np.float32)
            gmm_c_mean = float(i2_res["single_gmm_clean_mean"])
            gmm_n_mean = float(i2_res["single_gmm_noisy_mean"])
        elif pi_source_name == "pi_var":
            next_pi_np = np.asarray(i2_res["pi_var"], dtype=np.float32)
            gmm_c_mean = float(i2_res["mean_gmm_clean_mean"])
            gmm_n_mean = float(i2_res["mean_gmm_noisy_mean"])
        else:
            next_pi_np = np.asarray(i2_res["pi_mean"], dtype=np.float32)
            gmm_c_mean = float(i2_res["mean_gmm_clean_mean"])
            gmm_n_mean = float(i2_res["mean_gmm_noisy_mean"])

        sep_diag: Optional[Dict[str, Any]] = None
        if variant == "A-K1-G":
            gmm = i2_res["gmm_params"]
            raw_pi = next_pi_np.copy()
            next_pi_np, sep_s = apply_separation_gate(gmm, args.sep_gate_mode, args.sep_d_lo, args.sep_d_hi, args.sep_weight_thr)
            def pi_stats(values: np.ndarray) -> Dict[str, Any]:
                clean = ~noise_mask
                auc = None
                if 0 < int(noise_mask.sum()) < len(noise_mask):
                    auc = float(roc_auc_score(clean.astype(np.int32), values))
                return {"mean": float(np.mean(values)), "lt_05_fraction": float(np.mean(values < 0.5)), "roc_auc": auc, "clean_lt_05_fraction": float(np.mean(values[clean] < 0.5)) if np.any(clean) else None}
            sep_diag = {"gmm_means": gmm["means"], "gmm_vars": gmm["variances"], "gmm_weights": gmm["weights"], "sep_D": gmm["D"], "sep_s": sep_s, "fallback": gmm["fallback"], "pi_raw": pi_stats(raw_pi), "pi_applied": pi_stats(next_pi_np)}

        q_diag_raw_tensor = all_q_raw
        soft_zero_mean_fallback_count: Optional[int] = None
        if soft_active:
            q_diag_raw_tensor = compute_soft_bit_gate(all_q_raw, beta=float(args.soft_beta))
            soft_zero_mean_fallback_count = int((all_q_raw.mean(dim=1) <= 1e-12).sum().item())
        if bitwise_active:
            if is_rand_q:
                next_q_source, _ = permute_q_per_sample(all_q_raw, randq_gen)
            else:
                next_q_source = all_q_raw.detach().clone()
            next_q_tensor = (
                compute_soft_bit_gate(next_q_source, beta=float(args.soft_beta))
                if soft_active
                else next_q_source
            )
        else:
            next_q_tensor = torch.ones_like(all_q_raw)

        # 写入滞后一轮缓冲区（供 epoch + 1 使用）
        pi_buf.copy_(torch.from_numpy(next_pi_np).to(device=pi_buf.device, dtype=pi_buf.dtype))
        q_buf.copy_(next_q_tensor.to(device=q_buf.device, dtype=q_buf.dtype))
        q_raw_last_np = all_q_raw.detach().cpu().numpy().astype(np.float32)
        q_diag_raw_np = q_diag_raw_tensor.detach().cpu().numpy().astype(np.float32)
        next_q_active_np = q_buf.detach().cpu().numpy().astype(np.float32)
        snapshot_source_epoch = int(epoch)

        i2_diag = summarize_i2_diagnostics(
            i2_res=i2_res,
            noise_mask=noise_mask,
            label_inconsistent_mask=label_inconsistent_mask,
            probe_c1_check=probe_c1_check,
        )
        if sep_diag is not None:
            i2_diag["sep_gate"] = sep_diag
        i1_diag = summarize_i1_bitwise_diagnostics(
            pi_arr=next_pi_np,
            q_raw_arr=q_diag_raw_np,
            q_active_arr=next_q_active_np,
            momentum=float(args.momentum),
            noise_mask=noise_mask,
            label_inconsistent_mask=label_inconsistent_mask,
            mem_signs_before=mem_signs_before,
            mem_signs_after=mem_signs_after,
            zero_norm_epoch=int(i1_nce.zero_norm_events_epoch),
            zero_norm_total=int(i1_nce.zero_norm_events_total),
            is_rand_q=is_rand_q,
            soft_beta=float(args.soft_beta) if soft_active else None,
            soft_zero_mean_fallback_count=soft_zero_mean_fallback_count,
        )
        if soft_active:
            i1_diag["scalar_update_direction"] = i1_nce.scalar_direction_epoch_summary()

        return next_pi_np, gmm_c_mean, gmm_n_mean, i2_diag, i1_diag

    image_model_state_dict = {k: v.clone() for k, v in image_model.state_dict().items()}
    text_model_state_dict = {k: v.clone() for k, v in text_model.state_dict().items()}

    lr_schedu.step(start_epoch)
    for epoch in range(start_epoch, args.max_epochs):
        ep_t0 = time.time()
        a_warmup_active = bool(epoch < int(args.mcr_a_warmup))

        # 记录本轮训练实际生效的缓冲快照；PERM-pi 每轮至多置换一次且 warmup 不耗随机数
        warmup_or_off = bool((not i1_active) or a_warmup_active)
        applied_from_ep: Optional[int] = None if warmup_or_off else snapshot_source_epoch
        applied_pi_buf, perm_diag = prepare_applied_pi(
            pi_buf,
            perm_enabled=perm_pi_enabled,
            warmup=warmup_or_off,
            generator=perm_gen,
            permutation_seed=perm_seed,
            perm_seed_offset=int(args.perm_seed_offset),
            source_snapshot_from_epoch=applied_from_ep,
        )
        applied_pi_np = applied_pi_buf.detach().cpu().numpy().copy()
        if warmup_or_off:
            applied_q_np = np.ones((n_data, int(args.bit)), dtype=np.float32)
        else:
            applied_q_np = (
                q_buf.detach().cpu().numpy().copy()
                if bitwise_active
                else np.ones((n_data, int(args.bit)), dtype=np.float32)
            )

        mem_signs_before_ep: Optional[np.ndarray] = None
        if i1_active:
            with torch.no_grad():
                mem_signs_before_ep = (
                    contrast.memory.detach().sign().cpu().numpy().astype(np.int8)
                )

        train_loss_m, lc_m, lr_m = train_one_epoch(epoch)

        mem_signs_after_ep: Optional[np.ndarray] = None
        if i1_active:
            with torch.no_grad():
                mem_signs_after_ep = (
                    contrast.memory.detach().sign().cpu().numpy().astype(np.int8)
                )

        lr_schedu.step(epoch)
        i2t, t2i = test_epoch(epoch, is_eval=True)
        avg = (i2t + t2i) / 2.0
        if avg == best_acc:
            image_model_state_dict = {
                key: val.clone() for key, val in image_model.state_dict().items()
            }
            text_model_state_dict = {
                key: val.clone() for key, val in text_model.state_dict().items()
            }

        eff_mom_this_ep = 0.0 if (epoch < int(args.warmup_epoch)) else float(args.momentum)
        applied_budget_summary = summarize_i1_bitwise_diagnostics(
            pi_arr=(applied_pi_np if memory_gating else np.ones(n_data, dtype=np.float32)),
            q_raw_arr=applied_q_np,
            q_active_arr=applied_q_np,
            momentum=eff_mom_this_ep,
            noise_mask=noise_mask,
            label_inconsistent_mask=label_inconsistent_mask,
            is_rand_q=is_rand_q and (not a_warmup_active),
        )["write_budget_1_minus_m"]

        if i1_active:
            assert mem_signs_before_ep is not None and mem_signs_after_ep is not None
            next_pi_np, gmm_c_mean, gmm_n_mean, i2_diag, i1_diag = run_epoch_end_i1_traversal(
                epoch=epoch,
                mem_signs_before=mem_signs_before_ep,
                mem_signs_after=mem_signs_after_ep,
            )
        else:
            next_pi_np = np.ones(n_data, dtype=np.float32)
            gmm_c_mean = None
            gmm_n_mean = None
            i2_diag = None
            i1_diag = None

        ep_seconds = time.time() - ep_t0
        epoch_diagnostics.append(
            build_epoch_diag_entry(
                epoch=epoch,
                variant=variant,
                pi_source=pi_source_name,
                train_loss_mean=train_loss_m,
                lc_loss_mean=lc_m,
                lr_loss_mean=lr_m,
                epoch_seconds=ep_seconds,
                a_warmup_active=a_warmup_active,
                applied_snapshot_from_epoch=applied_from_ep,
                applied_pi_arr=applied_pi_np,
                applied_q_arr=applied_q_np,
                applied_write_budget_diag=applied_budget_summary,
                next_pi_arr=next_pi_np,
                gmm_clean_mean=gmm_c_mean,
                gmm_noisy_mean=gmm_n_mean,
                noise_mask=noise_mask,
                label_inconsistent_mask=label_inconsistent_mask,
                i2_diag=i2_diag,
                i1_diag=i1_diag,
                perm_diag=perm_diag,
            )
        )

    # 保存当前 pi/q 缓冲区与最后记忆库符号快照（M3 第 5.3 节要求）
    if i1_active:
        try:
            with torch.no_grad():
                final_mem_signs = contrast.memory.detach().sign().cpu().numpy().astype(np.int8)
                final_pi_np = pi_buf.detach().cpu().numpy().astype(np.float32)
                final_applied_pi_np = applied_pi_buf.detach().cpu().numpy().astype(np.float32)
                final_q_np = q_buf.detach().cpu().numpy().astype(np.float32)
            with open(out_dir / "i1_buffers_latest.npz", "wb") as f_buf:
                np.savez(
                    f_buf,
                    pi_buf=final_pi_np,
                    applied_pi_buf=final_applied_pi_np,
                    q_buf=final_q_np,
                    q_raw_last=q_raw_last_np,
                    memory_sign_snapshot=final_mem_signs,
                    snapshot_source_epoch=np.array(
                        [-1 if snapshot_source_epoch is None else int(snapshot_source_epoch)],
                        dtype=np.int64,
                    ),
                )
        except OSError as exc:
            print(f"[警告] 保存 I1 缓冲区快照失败 ({out_dir}): {exc}", file=sys.stderr)

    ckpt_path = os.path.join(args.ckpt_dir, "%s_%d_best_checkpoint.t7" % (args.arch, args.bit))
    if not args.no_save_ckpt and os.path.exists(ckpt_path):
        chp = torch.load(ckpt_path)
        final_eval_epoch = int(chp["epoch"])
    else:
        final_eval_epoch = int(best_epoch)

    image_model.load_state_dict(image_model_state_dict)
    text_model.load_state_dict(text_model_state_dict)
    test_epoch(final_eval_epoch, is_eval=False)
    summary_writer.close()

    return records, epoch_diagnostics, noise_mask, label_diagnostic


def main() -> None:
    args, upstream_passthrough = parse_all_args(sys.argv[1:])
    out_dir = Path(args.out).expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.exit(f"创建输出目录失败 ({out_dir}): {exc}")

    data_root = resolve_data_root(args.data_root)
    injected_modules = ensure_compat_modules()
    patch_torch_load_for_pt26()

    orig_cwd = Path.cwd()
    work_dir_str = tempfile.mkdtemp(prefix="i1_run_")
    work_dir = Path(work_dir_str).resolve()
    data_link = work_dir / "data"

    try:
        link_mode = link_data_directory(data_root, data_link)

        upstream_str = str(UPSTREAM)
        if upstream_str not in sys.path:
            sys.path.insert(0, upstream_str)
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{upstream_str}{os.pathsep}{old_pythonpath}" if old_pythonpath else upstream_str
        )

        os.chdir(work_dir)

        try:
            bar_show = importlib.import_module("utils.bar_show")
            if getattr(bar_show, "term_width", 80) > 200:
                setattr(bar_show, "term_width", 80)
        except Exception:
            pass

        dummy_holder = argparse.Namespace()
        apply_seed(dummy_holder, int(args.seed), bool(args.deterministic))

        t0 = time.time()
        records, epoch_diagnostics, noise_mask, label_diagnostic = run_training(args, out_dir)
        wall_seconds = round(time.time() - t0, 2)

        if len(records) < 2:
            sys.exit(f"评估函数调用次数异常（共 {len(records)} 次，预期至少 2 次）")

        test_i2t_rec = records[-2]
        test_t2i_rec = records[-1]
        epoch_records = build_epoch_records(records)

        for idx_ep, ep_rec in enumerate(epoch_records):
            if idx_ep < len(epoch_diagnostics):
                for k, v in epoch_diagnostics[idx_ep].items():
                    if k != "epoch":
                        ep_rec[k] = v

        best_val_epoch = -1
        best_val_avg = -1.0
        for ep_rec in epoch_records:
            if ep_rec["val_avg_official"] > best_val_avg:
                best_val_avg = float(ep_rec["val_avg_official"])
                best_val_epoch = int(ep_rec["epoch"])

        parsed_upstream_args = {k: getattr(args, k) for k in UPSTREAM_PARAM_KEYS if hasattr(args, k)}
        variant: I1Variant = args.variant
        eff_k = resolve_effective_k_views(variant, int(args.num_views))
        pi_source = resolve_pi_source_name(variant, bool(args.use_var))
        loss_weighting, memory_gating, perm_pi_enabled = resolve_dual_path_flags(variant)

        i1_hyperparams: Dict[str, Any] = {
            "variant": str(variant),
            "use_var": bool(args.use_var),
            "pi_source": str(pi_source),
            "loss_weighting": bool(loss_weighting),
            "memory_gating": bool(memory_gating),
            "perm_pi": bool(perm_pi_enabled),
            "bitwise_memory_gate": bool(variant in ("I1", "RAND-Q", "SOFT-Q", "SOFT-RAND-Q", "I1-V")),
            "rand_q_permutation": bool(variant in ("RAND-Q", "SOFT-RAND-Q")),
            "soft_bit_gate": bool(variant in ("SOFT-Q", "SOFT-RAND-Q")),
            "soft_beta": float(args.soft_beta),
            "probe_only_diagnostic": bool(variant == "probe"),
            "num_views_k": int(eff_k),
            "i2_lambda": float(args.i2_lambda),
            "i2_lambda_grid": [float(x) for x in args.i2_lambda_grid],
            "mcr_signal": str(args.mcr_signal),
            "mcr_a_warmup": int(args.mcr_a_warmup),
            "mcr_gamma": float(args.mcr_gamma),
            "mcr_delta": float(args.mcr_delta),
            "mcr_normalize": str(args.mcr_normalize),
            "mcr_rho_bar": float(args.mcr_rho_bar),
            "mcr_mask_rescale": bool(args.mcr_mask_rescale),
            "mcr_seed_offset": int(args.mcr_seed_offset),
            "probe_seed_offset": int(args.probe_seed_offset),
            "randq_seed_offset": int(args.randq_seed_offset),
            "perm_seed_offset": int(args.perm_seed_offset),
            "noise_rate": float(args.noise_rate),
            "noise_seed": int(args.noise_seed),
        }

        mean_epoch_sec = (
            round(float(np.mean([d["epoch_seconds"] for d in epoch_diagnostics])), 4)
            if epoch_diagnostics
            else 0.0
        )

        test_i2t_off = float(test_i2t_rec["map_official"])
        test_t2i_off = float(test_t2i_rec["map_official"])
        test_i2t_stb = float(test_i2t_rec["map_stable"])
        test_t2i_stb = float(test_t2i_rec["map_stable"])

        summary: Dict[str, Any] = {
            "seed": int(args.seed),
            "deterministic": bool(args.deterministic),
            "variant": str(variant),
            "noise_rate": float(args.noise_rate),
            "noise_seed": int(args.noise_seed),
            "n_noisy": int(np.sum(noise_mask)),
            "n_train": int(noise_mask.shape[0]),
            "label_inconsistency": label_diagnostic,
            "diagnostic_definitions": {
                "noise_mask": "文本被置换的位置，不等同于实际标签不一致",
                "pi_auc": "区分未置换与文本被置换样本的ROC-AUC",
                "pi_label_inconsistent_auc": "区分其余样本与置换后无共享正标签样本的ROC-AUC",
                "clean_low_pi_tail_lt_05": "干净组中 pi < 0.5 的工作阈值尾部比例（不解释为校准错误率）",
                "write_budget_1_minus_m": "主写入预算 mean(1 - m[i, l])，按干净、置换仍共享标签、实际标签不一致分层",
                "label_usage": "仅事后诊断，不参与训练、GMM或选模",
            },
            "i1_hyperparams": i1_hyperparams,
            "mcr_hyperparams": i1_hyperparams,
            "i1": {
                "variant": str(variant),
                "hyperparams": i1_hyperparams,
                "mean_epoch_seconds": mean_epoch_sec,
                "buffers_snapshot_file": "i1_buffers_latest.npz" if variant != "off" else None,
                "last_epoch_diagnostics": epoch_diagnostics[-1] if epoch_diagnostics else None,
                "epoch_diagnostics": epoch_diagnostics,
            },
            "mcr": {
                "variant": str(variant),
                "hyperparams": i1_hyperparams,
                "mean_epoch_seconds": mean_epoch_sec,
                "last_epoch_diagnostics": epoch_diagnostics[-1] if epoch_diagnostics else None,
                "epoch_diagnostics": epoch_diagnostics,
            },
            "data_root": str(data_root),
            "data_link_mode": link_mode,
            "injected_compat_modules": injected_modules,
            "upstream_args": upstream_passthrough,
            "parsed_args": parsed_upstream_args,
            "best_val_epoch": best_val_epoch,
            "best_val_avg_official": best_val_avg if best_val_epoch >= 0 else None,
            "test_i2t_official": test_i2t_off,
            "test_t2i_official": test_t2i_off,
            "test_avg_official": (test_i2t_off + test_t2i_off) / 2.0,
            "test_i2t_stable": test_i2t_stb,
            "test_t2i_stable": test_t2i_stb,
            "test_avg_stable": (test_i2t_stb + test_t2i_stb) / 2.0,
            "n_query": int(test_i2t_rec["n_query"]),
            "n_db": int(test_i2t_rec["n_db"]),
            "epoch_records": epoch_records,
            "eval_records": records,
            "wall_seconds": wall_seconds,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

        if variant == "A-K1-G":
            summary["sep_gate"] = {"mode": args.sep_gate_mode, "d_lo": args.sep_d_lo, "d_hi": args.sep_d_hi, "weight_thr": args.sep_weight_thr}
        summary_path = out_dir / "summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f_sum:
                json.dump(summary, f_sum, indent=2, ensure_ascii=False)
        except OSError as exc:
            sys.exit(f"写 summary.json 失败 ({summary_path}): {exc}")

        brief = {
            k: v
            for k, v in summary.items()
            if k not in ("eval_records", "epoch_records", "parsed_args", "mcr", "i1")
        }
        print("SUMMARY", json.dumps(brief, ensure_ascii=False))
    finally:
        try:
            os.chdir(orig_cwd)
        except OSError:
            pass
        remove_data_link(data_link)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
