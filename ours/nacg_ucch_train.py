"""NACG-Hash 在 UCCH 主线上的 GPU 常驻零误差训练与评估入口（支持 off / NC-MRCH / UCCH-NACG / RCH-v2..v6）。

设计特点：
  1. 当 ``--variant off`` 时，严格走上游 UCCH 原始公式、初始化顺序、RNG 消耗与损失计算，
     同时利用 GPU 常驻批采样与 GPU MAP@ALL 引擎实现与 ``i1_train.py --variant off`` 逐位等价（0.0 误差）；
  2. 当 ``--variant NC-MRCH``（纯微创手术版，``--lambda_graph 0``）时：
     - 训练前先执行无模型污染的冷启动 ODNC 检测，``hat_eta_0 == 0`` 下自动触发精确 0.0 无损退化；
     - ``hat_eta_0 > 0`` 时，使用 ``NACGNCEAverage`` 从 Epoch 0 起以干净桥接正键净化记忆库，
       使用 ``nacg_ucch_contrastive_loss``（``normalize='sum_w'`` 梯度无萎缩、干净正锚替换），不加额外图 MSE；
  3. 当 ``--variant UCCH-NACG``（完整版，默认 ``--lambda_graph 0.2``）时：
     - 叠加多标签假阴性豁免（``false_neg_sim > 0.25``）与轻量化正向近邻图结构辅助损失；
  4. 当 ``--variant RCH-v4`` / ``RCH-v5`` 时保留各自迭代完整实验分支以便可复现归档；
  5. 当 ``--variant RCH-v6`` / ``RCH-v6-con``（默认 ``lambda_graph=0.35, alpha=0.65`` / ``lambda_graph=0.22, alpha=0.70``）时：
     - 对称 ``MCRNCEAverage``（Epoch 0 ``w=1`` + ``r_0^2`` 干净记忆门控，后续轮次 ``w_pair = 0.92*r + 0.08`` 且 ``normalize='sum_w'`` 消除损失尺度萎缩）；
     - 锐化干净锚点桥接 + 零均值有符号二跳亲和轮廓（消除 ``F.relu`` 假阳性正偏置）+ 模态自适应正向图回归损失与在线桥接刷新。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
import torch.optim as optim  # type: ignore[import-not-found]
from torch.nn.utils.clip_grad import clip_grad_norm  # type: ignore[import-not-found]
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler  # type: ignore[import-not-found]

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent
UPSTREAM = BASELINE_DIR / "upstream"

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baselines.protocol import (  # type: ignore[import-not-found]
    apply_training_noise,
    evaluate_codes,
    load_protocol_data,
    save_noise_data,
    write_json_atomic,
)
from ours.mcr import MCRNCEAverage, weighted_nce_softmax_loss  # type: ignore[import-not-found]
from ours.nacg import (  # type: ignore[import-not-found]
    NACGAnchorState,
    NACGNCEAverage,
    RCHNCEAverage,
    RCHv3NCEAverage,
    RCHv4NCEAverage,
    RCHv5NCEAverage,
    build_nacg_anchor_state,
    compute_centered_clean_graph,
    compute_clean_cross_sim,
    compute_rch_sample_weights,
    compute_v5_clean_affinity_and_loss,
    compute_v6_clean_affinity_and_loss,
    compute_zscore_clean_graph,
    gather_bridged_features,
    nacg_positive_graph_loss,
    nacg_ucch_contrastive_loss,
    rch_clean_graph_loss,
    rch_contrastive_loss,
    rch_v3_contrastive_loss,
    rch_v3_neighbor_nce_loss,
    rch_v4_positive_graph_loss,
    rch_v5_contrastive_loss,
    update_nacg_epoch_reliability,
)
from run_ucch import (  # type: ignore[import-not-found]
    apply_seed,
    ensure_compat_modules,
    patch_torch_load_for_pt26,
    resolve_data_root,
)


class IndexedPairDataset(Dataset):
    def __init__(self, images: np.ndarray, texts: np.ndarray, labels: np.ndarray) -> None:
        self.images = images
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> Tuple[int, List[np.ndarray], List[np.ndarray], np.ndarray]:
        return index, [self.images[index]], [self.texts[index]], self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", required=True, choices=["mirflickr25k_fea", "nus_wide_tc10_fea", "mscoco_fea"])
    parser.add_argument("--data_root", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--variant",
        default="UCCH-NACG",
        choices=[
            "off",
            "NC-MRCH",
            "UCCH-NACG",
            "RCH-v2-base",
            "RCH-v2",
            "RCH-v3-sym",
            "RCH-v3",
            "RCH-v4",
            "RCH-v4-ablate",
            "RCH-v4-nog",
            "RCH-v5-sym",
            "RCH-v5",
            "RCH-v5-nog",
            "RCH-v6-con",
            "RCH-v6",
            "RCH-v6-nog",
            "RCH-v6-stat",
            "RCH-v6-gonly",
            "RCH-v6-fnonly",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_rate", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--bit", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--num_hiden_layers", default=[3, 2], nargs="+", type=int)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--shift", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--ls", default="linear", choices=["linear", "cos"])
    parser.add_argument("--warmup_epoch", type=int, default=1)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--T", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--map_query_chunk_size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_true")
    parser.add_argument("--dump_codes", action="store_true")
    # NACG 专属超参（预设固定，不在测试集调参）
    parser.add_argument("--lambda_graph", type=float, default=None)
    parser.add_argument("--nacg_anchors", type=int, default=512)
    parser.add_argument("--nacg_top_k", type=int, default=16)
    parser.add_argument("--nacg_tau_bridge", type=float, default=0.07)
    parser.add_argument("--nacg_tau_null", type=float, default=0.12)
    parser.add_argument("--nacg_force_one", action="store_true")
    parser.add_argument("--rch_gamma_r", type=float, default=2.0)
    parser.add_argument("--rch_delta", type=float, default=0.02)
    parser.add_argument("--rch_pos_quantile", type=float, default=0.85)
    parser.add_argument("--rch_stretch", type=float, default=1.8)
    parser.add_argument("--rch_v3_delta", type=float, default=0.10)
    parser.add_argument("--rch_v3_top_k", type=int, default=6)
    parser.add_argument("--rch_v3_z_min", type=float, default=1.35)
    parser.add_argument("--rch_v3_z_fn_thr", type=float, default=1.25)
    parser.add_argument("--rch_v3_fn_z", type=float, default=0.80)
    parser.add_argument("--rch_v3_tau_g", type=float, default=0.50)
    parser.add_argument("--rch_v4_twin_start_epoch", type=int, default=1)
    parser.add_argument("--rch_v4_z_twin_thr", type=float, default=1.05)
    parser.add_argument("--rch_v4_z_nce_fn_thr", type=float, default=1.00)
    parser.add_argument("--rch_v4_z_fn_thr", type=float, default=1.00)
    parser.add_argument("--rch_v4_z_min", type=float, default=0.80)
    parser.add_argument("--rch_v4_target_base", type=float, default=0.45)
    parser.add_argument("--rch_v4_target_slope", type=float, default=0.20)
    parser.add_argument("--rch_v4_target_max", type=float, default=0.85)
    parser.add_argument("--rch_v4_quant_mix", type=float, default=0.25)
    parser.add_argument("--rch_v5_twin_start_epoch", type=int, default=1)
    parser.add_argument("--rch_v5_z_twin_thr", type=float, default=1.15)
    parser.add_argument("--rch_v5_sym_delta", type=float, default=0.05)
    parser.add_argument("--rch_v5_fn_thr", type=float, default=0.25)
    parser.add_argument("--rch_v5_stretch", type=float, default=1.25)
    parser.add_argument("--rch_v5_ho_weight", type=float, default=0.30)
    parser.add_argument("--rch_v5_upper_weight", type=float, default=0.10)
    parser.add_argument("--rch_v6_sym_delta", type=float, default=0.08)
    parser.add_argument("--rch_v6_fn_thr", type=float, default=0.25)
    parser.add_argument("--rch_v6_ho_weight", type=float, default=0.25)
    parser.add_argument("--rch_v6_bridge_sharpen", type=float, default=None)
    args = parser.parse_args()
    if args.alpha is None:
        args.alpha = (
            0.65
            if args.variant in ("RCH-v6", "RCH-v6-nog", "RCH-v6-stat", "RCH-v6-gonly", "RCH-v6-fnonly")
            else 0.70
        )
    if args.rch_v6_bridge_sharpen is None:
        args.rch_v6_bridge_sharpen = (
            1.0 if args.variant in ("RCH-v6-stat", "RCH-v6-gonly", "RCH-v6-fnonly") else 2.5
        )
    if args.lambda_graph is None:
        if args.variant in ("off", "NC-MRCH", "RCH-v2-base", "RCH-v4-nog", "RCH-v5-nog", "RCH-v6-nog", "RCH-v6-fnonly"):
            args.lambda_graph = 0.0
        elif args.variant == "RCH-v2":
            args.lambda_graph = 0.35
        elif args.variant == "RCH-v3-sym":
            args.lambda_graph = 0.30
        elif args.variant == "RCH-v3":
            args.lambda_graph = 0.45
        elif args.variant == "RCH-v4":
            args.lambda_graph = 0.80
        elif args.variant == "RCH-v4-ablate":
            args.lambda_graph = 0.40
        elif args.variant == "RCH-v5-sym":
            args.lambda_graph = 0.25
        elif args.variant == "RCH-v5":
            args.lambda_graph = 0.35
        elif args.variant == "RCH-v6-con":
            args.lambda_graph = 0.22
        elif args.variant in ("RCH-v6", "RCH-v6-stat", "RCH-v6-gonly"):
            args.lambda_graph = 0.35
        else:
            args.lambda_graph = 0.20
    if args.variant in ("off", "NC-MRCH", "RCH-v2-base"):
        args.lambda_graph = 0.0
    if not 0.0 <= float(args.noise_rate) <= 1.0:
        parser.error("noise_rate 必须位于 [0, 1]")
    return args


def _encode_codes(
    model: nn.Module,
    values: Any,
    batch_size: int,
    device: torch.device,
    normalize_continuous: bool = False,
    limit: Optional[int] = None,
) -> Any:
    model.eval()
    n_total = int(values.shape[0]) if limit is None else min(int(limit), int(values.shape[0]))
    is_gpu_tensor = isinstance(values, torch.Tensor)
    chunks = []
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            if is_gpu_tensor:
                batch = values[start:end]
            else:
                batch = torch.from_numpy(values[start:end]).to(device=device, dtype=torch.float32)
            out = model(batch.float())
            chunks.append(F.normalize(out, dim=1) if normalize_continuous else torch.sign(out).to(torch.int8))
    joined = torch.cat(chunks, dim=0)
    if normalize_continuous:
        return joined
    return joined.cpu().numpy()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("正式 NACG 实验要求 CUDA，但当前不可用")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data_root)

    ensure_compat_modules()
    patch_torch_load_for_pt26()
    upstream_str = str(UPSTREAM)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)
    old_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{upstream_str}{os.pathsep}{old_pp}" if old_pp else upstream_str

    models_mod = importlib.import_module("nets")
    utils_mod = importlib.import_module("src.utils")
    nce_avg_mod = importlib.import_module("NCE.NCEAverage")
    nce_crit_mod = importlib.import_module("NCE.NCECriterion")
    image_net_cls: Any = getattr(models_mod, "ImageNet")
    text_net_cls: Any = getattr(models_mod, "TextNet")
    NCEAverage: Any = getattr(nce_avg_mod, "NCEAverage")
    NCESoftmaxLoss: Any = getattr(nce_crit_mod, "NCESoftmaxLoss")

    # 严格复用上游种子设置序列（不修改 upstream）
    dummy_holder = argparse.Namespace()
    apply_seed(dummy_holder, int(args.seed), bool(args.deterministic))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    data = load_protocol_data(data_root, args.data_name)
    noise = apply_training_noise(data, float(args.noise_rate), int(args.noise_seed))
    save_noise_data(out_dir, noise)

    image_model: nn.Module = image_net_cls(
        y_dim=int(data.images.shape[1]),
        bit=int(args.bit),
        hiden_layer=int(args.num_hiden_layers[0]),
    ).to(device)
    text_model: nn.Module = text_net_cls(
        y_dim=int(data.texts.shape[1]),
        bit=int(args.bit),
        hiden_layer=int(args.num_hiden_layers[1]),
    ).to(device)

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    if args.optimizer == "SGD":
        optimizer: optim.Optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)

    if args.ls == "cos":
        lr_schedu: Any = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.1)

    criterion = utils_mod.ContrastiveLoss(args.margin, shift=args.shift)
    n_data = int(data.n_train)
    criterion_contrast = NCESoftmaxLoss()
    # 注意：原始 NCEAverage.__init__ 固定调用 AliasMethod.cuda()；CUDA 路径保持 100% 原生，CPU 模式临时屏蔽 .cuda() 以兼容纯 CPU 单测
    if device.type == "cuda":
        contrast = NCEAverage(args.bit, n_data, args.K, args.T, args.momentum)
        contrast = contrast.to(device)
        criterion_contrast = criterion_contrast.to(device)
    else:
        alias_mod = importlib.import_module("NCE.alias_multinomial")
        alias_cls: Any = getattr(alias_mod, "AliasMethod")
        orig_alias_cuda = alias_cls.cuda
        alias_cls.cuda = lambda self_a: self_a
        try:
            contrast = NCEAverage(args.bit, n_data, args.K, args.T, args.momentum)
        finally:
            alias_cls.cuda = orig_alias_cuda

    nacg_nce = NACGNCEAverage(contrast)
    rch_nce = RCHNCEAverage(contrast)
    mcr_nce = MCRNCEAverage(contrast)
    rch_v3_nce = RCHv3NCEAverage(contrast)
    rch_v4_nce = RCHv4NCEAverage(contrast)
    rch_v5_nce = RCHv5NCEAverage(contrast)

    use_gpu_resident = device.type == "cuda" and os.environ.get("DISABLE_NACG_GPU_RESIDENT") != "1"
    val_size = min(2000, n_data)
    eval_encode_limit = min(n_data, ((val_size + args.eval_batch_size - 1) // args.eval_batch_size) * args.eval_batch_size)
    if use_gpu_resident:
        train_images_t = torch.as_tensor(data.images, device=device, dtype=torch.float32).contiguous()
        train_texts_t = torch.as_tensor(noise.train_texts, device=device, dtype=torch.float32).contiguous()
        val_images_t = train_images_t[:eval_encode_limit]
        val_texts_t = torch.as_tensor(data.texts[:eval_encode_limit], device=device, dtype=torch.float32).contiguous()
        val_labels = data.labels[:val_size]
        train_loader = None
    else:
        train_images_t = torch.as_tensor(data.images, device=device, dtype=torch.float32).contiguous()
        train_texts_t = torch.as_tensor(noise.train_texts, device=device, dtype=torch.float32).contiguous()
        val_images_t = data.images[:eval_encode_limit]
        val_texts_t = data.texts[:eval_encode_limit]
        val_labels = data.labels[:val_size]
        train_loader = DataLoader(
            IndexedPairDataset(data.images, noise.train_texts, data.labels),
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

    is_v2 = args.variant in ("RCH-v2-base", "RCH-v2")
    is_v3 = args.variant in ("RCH-v3-sym", "RCH-v3")
    is_v4 = args.variant in ("RCH-v4", "RCH-v4-ablate", "RCH-v4-nog")
    is_v5 = args.variant in ("RCH-v5-sym", "RCH-v5", "RCH-v5-nog")
    is_v6 = args.variant in ("RCH-v6-con", "RCH-v6", "RCH-v6-nog", "RCH-v6-stat", "RCH-v6-gonly", "RCH-v6-fnonly")
    refresh_bridges_v6 = args.variant in ("RCH-v6", "RCH-v6-con")
    store_full_norm = not (
        is_v2
        or is_v3
        or is_v4
        or (args.variant == "RCH-v6-nog" and float(args.lambda_graph) <= 0.0)
    )
    nacg_active = args.variant != "off"
    nacg_state: Optional[NACGAnchorState] = None
    if nacg_active:
        nacg_state = build_nacg_anchor_state(
            images_gpu=train_images_t,
            texts_gpu=train_texts_t,
            seed=int(args.seed),
            num_anchors=int(args.nacg_anchors),
            top_k_bridge=int(args.nacg_top_k),
            tau_bridge=float(args.nacg_tau_bridge),
            tau_null=float(args.nacg_tau_null),
            noise_mask=noise.mask,
            store_full_norm=store_full_norm,
            self_consistent_storey=is_v4,
        )
        if args.nacg_force_one:
            nacg_state.hat_eta_0 = 0.0
            nacg_state.r_0.fill_(1.0)
            nacg_state.r.fill_(1.0)

    is_full_nacg = args.variant == "UCCH-NACG"
    lambda_graph = float(args.lambda_graph)

    best_acc = 0.0
    best_epoch = 0
    best_img_state = {k: v.detach().cpu().clone() for k, v in image_model.state_dict().items()}
    best_txt_state = {k: v.detach().cpu().clone() for k, v in text_model.state_dict().items()}
    epoch_records: List[Dict[str, Any]] = []

    lr_schedu.step(0)
    for epoch in range(int(args.max_epochs)):
        ep_t0 = time.perf_counter()
        image_model.train()
        text_model.train()

        use_nacg_this_ep = False
        use_graph_this_ep = False
        use_v2_rel = False
        use_v2_graph = False
        use_v3_rel = False
        use_v3_zgraph = False
        use_v3_nbr_loss = False
        use_v4_active = False
        use_v4_nbr_loss = False
        use_v5_active = False
        use_v5_graph = False
        use_v6_active = False
        use_v6_graph = False
        if nacg_state is not None and not args.nacg_force_one:
            is_noisy_gate = not bool(torch.all(nacg_state.r == 1.0).item())
            if is_v6:
                use_v6_active = True
                use_v6_graph = bool(lambda_graph > 0.0)
            elif is_v5:
                use_v5_active = True
                use_v5_graph = bool(lambda_graph > 0.0)
            elif is_v4:
                use_v4_active = True
                use_v4_nbr_loss = bool(lambda_graph > 0.0)
            elif is_v3:
                use_v3_rel = bool(is_noisy_gate and epoch >= int(args.warmup_epoch))
                # [P1-6]: 即使 --lambda_graph 0 也必须计算批内 Z-score 干净亲和图，供 NCE 孪生路由与排序假阴性屏蔽使用
                use_v3_zgraph = True
                use_v3_nbr_loss = bool(lambda_graph > 0.0)
            elif is_v2:
                use_v2_rel = is_noisy_gate
                use_v2_graph = bool(args.variant == "RCH-v2" and lambda_graph > 0.0)
            else:
                use_nacg_this_ep = is_noisy_gate
                use_graph_this_ep = bool(is_full_nacg and lambda_graph > 0.0)

        anc_imgs = None
        anc_txts = None
        anc_hx = None
        anc_hy = None
        anc_norm_img = None
        anc_norm_txt = None
        if nacg_state is not None and (use_nacg_this_ep or use_graph_this_ep):
            anc_imgs = train_images_t.index_select(0, nacg_state.anchor_idx)
            anc_txts = train_texts_t.index_select(0, nacg_state.anchor_idx)
            anc_norm_img = nacg_state.norm_images.index_select(0, nacg_state.anchor_idx)
            anc_norm_txt = nacg_state.norm_texts.index_select(0, nacg_state.anchor_idx)

        if use_gpu_resident:
            torch.empty((), dtype=torch.int64).random_()
            batch_iter: Any = BatchSampler(
                RandomSampler(range(n_data)),
                batch_size=args.train_batch_size,
                drop_last=True,
            )
            n_batches_est = max(n_data // args.train_batch_size, 1)
        else:
            assert train_loader is not None
            batch_iter = train_loader
            n_batches_est = max(len(train_loader), 1)

        refresh_interval = max(1, n_batches_est // 8)
        losses: List[float] = []
        for batch_step, batch_item in enumerate(batch_iter):
            if use_gpu_resident:
                idx_t = torch.as_tensor(batch_item, device=device, dtype=torch.long)
                imgs = train_images_t.index_select(0, idx_t)
                txts = train_texts_t.index_select(0, idx_t)
            else:
                idx_cpu, imgs_list, txts_list, _ = batch_item
                idx_t = idx_cpu.to(device=device, dtype=torch.long).view(-1)
                imgs = imgs_list[0].to(device=device, dtype=torch.float32)
                txts = txts_list[0].to(device=device, dtype=torch.float32)

            if use_nacg_this_ep and anc_imgs is not None and anc_txts is not None and (batch_step % refresh_interval == 0):
                was_training_i = image_model.training
                was_training_t = text_model.training
                image_model.eval()
                text_model.eval()
                with torch.no_grad():
                    anc_hx = F.normalize(image_model(anc_imgs), dim=1)
                    anc_hy = F.normalize(text_model(anc_txts.float()), dim=1)
                if was_training_i:
                    image_model.train()
                if was_training_t:
                    text_model.train()

            im_out = image_model(imgs)
            txt_out = text_model(txts.float())

            if (
                not nacg_active
                or args.nacg_force_one
                or (is_v6 and not use_v6_active)
                or (is_v5 and not use_v5_active)
                or (is_v4 and not use_v4_active)
                or (is_v3 and not use_v3_rel and not use_v3_zgraph)
                or (is_v2 and not use_v2_rel and not use_v2_graph)
                or ((not is_v2) and (not is_v3) and (not is_v4) and (not is_v5) and (not is_v6) and not use_nacg_this_ep and not use_graph_this_ep)
            ):
                out_l, out_ab = contrast(im_out, txt_out, idx_t, epoch=epoch - args.warmup_epoch)
                l_loss = criterion_contrast(out_l)
                ab_loss = criterion_contrast(out_ab)
                lc_val = l_loss + ab_loss
                lr_val = criterion(im_out, txt_out)
                loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha)
            elif is_v6:
                assert nacg_state is not None
                r_b = nacg_state.r.index_select(0, idx_t).to(device)
                v_b = nacg_state.valid_mask.index_select(0, idx_t).to(device)
                nbr_idx_i_b = nacg_state.nbr_idx_I.index_select(0, idx_t).to(device)
                nbr_w_i_b = nacg_state.nbr_w_I.index_select(0, idx_t).to(device)
                nbr_idx_t_b = nacg_state.nbr_idx_T.index_select(0, idx_t).to(device)
                nbr_w_t_b = nacg_state.nbr_w_T.index_select(0, idx_t).to(device)
                is_warmup_ep = bool(epoch < int(args.warmup_epoch))
                is_r_clean_ep = bool(torch.all(r_b == 1.0).item())

                need_v6_affinity = bool(use_v6_graph or args.variant == "RCH-v6-fnonly")
                if not need_v6_affinity:
                    l_graph_v6 = (im_out * 0.0).sum() + (txt_out * 0.0).sum()
                    sim_it_v6: Optional[torch.Tensor] = None
                else:
                    l_graph_v6, sim_it_v6 = compute_v6_clean_affinity_and_loss(
                        im_codes=im_out,
                        txt_codes=txt_out,
                        imgs=imgs,
                        txts=txts,
                        r_batch=r_b,
                        valid_batch=v_b,
                        nbr_idx_i=nbr_idx_i_b,
                        nbr_w_i=nbr_w_i_b,
                        nbr_idx_t=nbr_idx_t_b,
                        nbr_w_t=nbr_w_t_b,
                        state=nacg_state,
                        ho_weight=float(args.rch_v6_ho_weight),
                        bridge_sharpen=float(args.rch_v6_bridge_sharpen),
                    )
                fn_sim_v6: Optional[torch.Tensor] = None if args.variant == "RCH-v6-gonly" else sim_it_v6

                if is_r_clean_ep:
                    out_l, out_ab = contrast(im_out, txt_out, idx_t, epoch=epoch - int(args.warmup_epoch))
                    l_loss = criterion_contrast(out_l)
                    ab_loss = criterion_contrast(out_ab)
                    w_pair_v6: Optional[torch.Tensor] = None
                elif is_warmup_ep:
                    out_l, out_ab = mcr_nce(
                        im_out,
                        txt_out,
                        idx_t,
                        epoch=epoch - int(args.warmup_epoch),
                        pi=r_b.pow(2),
                    )
                    l_loss = criterion_contrast(out_l)
                    ab_loss = criterion_contrast(out_ab)
                    w_pair_v6 = None
                else:
                    out_l, out_ab = mcr_nce(
                        im_out,
                        txt_out,
                        idx_t,
                        epoch=epoch - int(args.warmup_epoch),
                        pi=r_b,
                    )
                    w_pair_v6 = compute_rch_sample_weights(
                        r_b,
                        gamma_r=1.0,
                        delta=float(args.rch_v6_sym_delta),
                    )
                    l_loss = weighted_nce_softmax_loss(out_l, w_pair_v6, normalize="sum_w")
                    ab_loss = weighted_nce_softmax_loss(out_ab, w_pair_v6, normalize="sum_w")

                lc_val = l_loss + ab_loss
                lr_val = rch_v5_contrastive_loss(
                    im=im_out,
                    s=txt_out,
                    w_pair=w_pair_v6,
                    margin=float(args.margin),
                    shift=float(args.shift),
                    false_neg_sim=fn_sim_v6,
                    fn_thr=float(args.rch_v6_fn_thr),
                    normalize="sum_w",
                )
                loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha)
                if use_v6_graph:
                    loss = loss + lambda_graph * l_graph_v6
            elif is_v5:
                assert nacg_state is not None
                r_b = nacg_state.r.index_select(0, idx_t).to(device)
                v_b = nacg_state.valid_mask.index_select(0, idx_t).to(device)
                nbr_idx_i_b = nacg_state.nbr_idx_I.index_select(0, idx_t).to(device)
                nbr_w_i_b = nacg_state.nbr_w_I.index_select(0, idx_t).to(device)
                nbr_idx_t_b = nacg_state.nbr_idx_T.index_select(0, idx_t).to(device)
                nbr_w_t_b = nacg_state.nbr_w_T.index_select(0, idx_t).to(device)
                is_warmup_ep = bool(epoch < int(args.warmup_epoch))
                is_r_clean_ep = bool(torch.all(r_b == 1.0).item())

                l_graph_v5, sim_it_v5 = compute_v5_clean_affinity_and_loss(
                    im_codes=im_out,
                    txt_codes=txt_out,
                    imgs=imgs,
                    txts=txts,
                    r_batch=r_b,
                    valid_batch=v_b,
                    nbr_idx_i=nbr_idx_i_b,
                    nbr_w_i=nbr_w_i_b,
                    nbr_idx_t=nbr_idx_t_b,
                    nbr_w_t=nbr_w_t_b,
                    state=nacg_state,
                    stretch=float(args.rch_v5_stretch),
                    ho_weight=float(args.rch_v5_ho_weight),
                    upper_weight=float(args.rch_v5_upper_weight),
                )

                if args.variant == "RCH-v5-sym":
                    if is_r_clean_ep:
                        out_l, out_ab = contrast(im_out, txt_out, idx_t, epoch=epoch - int(args.warmup_epoch))
                        l_loss = criterion_contrast(out_l)
                        ab_loss = criterion_contrast(out_ab)
                        w_pair: Optional[torch.Tensor] = None
                    elif is_warmup_ep:
                        out_l, out_ab = mcr_nce(
                            im_out,
                            txt_out,
                            idx_t,
                            epoch=epoch - int(args.warmup_epoch),
                            pi=r_b.pow(2),
                        )
                        l_loss = criterion_contrast(out_l)
                        ab_loss = criterion_contrast(out_ab)
                        w_pair = None
                    else:
                        out_l, out_ab = mcr_nce(
                            im_out,
                            txt_out,
                            idx_t,
                            epoch=epoch - int(args.warmup_epoch),
                            pi=r_b,
                        )
                        w_pair = compute_rch_sample_weights(
                            r_b,
                            gamma_r=1.0,
                            delta=float(args.rch_v5_sym_delta),
                        )
                        l_loss = weighted_nce_softmax_loss(out_l, w_pair, normalize="n")
                        ab_loss = weighted_nce_softmax_loss(out_ab, w_pair, normalize="n")
                else:
                    z_ii_v5: Optional[torch.Tensor] = None
                    z_tt_v5: Optional[torch.Tensor] = None
                    if (not is_r_clean_ep) and (epoch >= int(args.rch_v5_twin_start_epoch)):
                        z_ii_v5, z_tt_v5, _, _, _, _ = compute_zscore_clean_graph(
                            imgs,
                            txts,
                            r_b,
                            v_b,
                            nacg_state,
                        )
                    out_l, out_ab, w_l, w_ab = rch_v5_nce(
                        im_out,
                        txt_out,
                        idx_t,
                        epoch=epoch - int(args.warmup_epoch),
                        r_batch=r_b,
                        valid_batch=v_b,
                        z_ii=z_ii_v5,
                        z_tt=z_tt_v5,
                        z_twin_thr=float(args.rch_v5_z_twin_thr),
                    )
                    l_loss = weighted_nce_softmax_loss(out_l, w_l, normalize="n")
                    ab_loss = weighted_nce_softmax_loss(out_ab, w_ab, normalize="n")
                    w_pair = None if (is_warmup_ep or is_r_clean_ep) else r_b.pow(2)

                lc_val = l_loss + ab_loss
                lr_val = rch_v5_contrastive_loss(
                    im=im_out,
                    s=txt_out,
                    w_pair=w_pair,
                    margin=float(args.margin),
                    shift=float(args.shift),
                    false_neg_sim=sim_it_v5,
                    fn_thr=float(args.rch_v5_fn_thr),
                )
                loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha)
                if use_v5_graph:
                    loss = loss + lambda_graph * l_graph_v5
            elif is_v4:
                assert nacg_state is not None
                r_state_b = nacg_state.r.index_select(0, idx_t).to(device)
                v_b = nacg_state.valid_mask.index_select(0, idx_t).to(device)
                z_ii, z_tt, z_it, om_ii, om_tt, om_it = compute_zscore_clean_graph(
                    imgs,
                    txts,
                    r_state_b,
                    v_b,
                    nacg_state,
                )
                use_twins_now = bool(epoch >= int(args.rch_v4_twin_start_epoch))
                out_l, out_ab, w_l, w_ab = rch_v4_nce(
                    im_out,
                    txt_out,
                    idx_t,
                    epoch=epoch - int(args.warmup_epoch),
                    r_all=nacg_state.r,
                    px_norm=nacg_state.px_norm,
                    py_norm=nacg_state.py_norm,
                    valid_mask=nacg_state.valid_mask,
                    z_ii=z_ii if use_twins_now else None,
                    z_tt=z_tt if use_twins_now else None,
                    z_twin_thr=float(args.rch_v4_z_twin_thr),
                    z_nce_fn_thr=float(args.rch_v4_z_nce_fn_thr),
                )
                l_loss = weighted_nce_softmax_loss(out_l, w_l, normalize="n")
                ab_loss = weighted_nce_softmax_loss(out_ab, w_ab, normalize="n")
                lc_val = l_loss + ab_loss
                w_lr_v4 = None if epoch < int(args.warmup_epoch) else 0.5 * (w_l + w_ab)
                lr_val = rch_v3_contrastive_loss(
                    im=im_out,
                    s=txt_out,
                    w_batch=w_lr_v4,
                    margin=float(args.margin),
                    shift=float(args.shift),
                    false_neg_z=z_it,
                    false_neg_conf=om_it,
                    z_fn_thr=float(args.rch_v4_z_fn_thr),
                )
                loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha)
                if use_v4_nbr_loss:
                    l_pos_g = rch_v4_positive_graph_loss(
                        im=im_out,
                        s=txt_out,
                        z_ii=z_ii,
                        z_tt=z_tt,
                        z_it=z_it,
                        omega_ii=om_ii,
                        omega_tt=om_tt,
                        omega_it=om_it,
                        z_min=float(args.rch_v4_z_min),
                        target_base=float(args.rch_v4_target_base),
                        target_slope=float(args.rch_v4_target_slope),
                        target_max=float(args.rch_v4_target_max),
                        quant_mix=float(args.rch_v4_quant_mix),
                    )
                    loss = loss + lambda_graph * l_pos_g
            elif is_v3:
                assert nacg_state is not None
                r_state_b = nacg_state.r.index_select(0, idx_t).to(device)
                v_b = nacg_state.valid_mask.index_select(0, idx_t).to(device)
                z_ii = z_tt = z_it = om_ii = om_tt = om_it = None
                if use_v3_zgraph:
                    z_ii, z_tt, z_it, om_ii, om_tt, om_it = compute_zscore_clean_graph(
                        imgs,
                        txts,
                        r_state_b,
                        v_b,
                        nacg_state,
                    )
                if use_v3_rel:
                    w_b = compute_rch_sample_weights(
                        r_state_b,
                        gamma_r=1.0,
                        delta=float(args.rch_v3_delta),
                    )
                    if args.variant == "RCH-v3":
                        out_l, out_ab = rch_v3_nce(
                            im_out,
                            txt_out,
                            idx_t,
                            epoch=epoch - args.warmup_epoch,
                            r_batch=r_state_b,
                            z_ii=z_ii,
                            valid_batch=v_b,
                            z_tt=z_tt,
                        )
                    else:
                        out_l, out_ab = mcr_nce(
                            im_out,
                            txt_out,
                            idx_t,
                            epoch=epoch - args.warmup_epoch,
                            pi=r_state_b,
                        )
                    l_loss = weighted_nce_softmax_loss(out_l, w_b, normalize="n")
                    ab_loss = weighted_nce_softmax_loss(out_ab, w_b, normalize="n")
                    lc_val = l_loss + ab_loss
                else:
                    w_b = torch.ones(int(idx_t.size(0)), device=device, dtype=im_out.dtype)
                    out_l, out_ab = contrast(im_out, txt_out, idx_t, epoch=epoch - args.warmup_epoch)
                    l_loss = criterion_contrast(out_l)
                    ab_loss = criterion_contrast(out_ab)
                    lc_val = l_loss + ab_loss

                lr_val = rch_v3_contrastive_loss(
                    im=im_out,
                    s=txt_out,
                    w_batch=w_b,
                    margin=float(args.margin),
                    shift=float(args.shift),
                    false_neg_z=z_it,
                    false_neg_conf=om_it,
                    z_fn_thr=float(args.rch_v3_z_fn_thr),
                )
                loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha)
                if use_v3_nbr_loss and z_ii is not None and z_tt is not None and z_it is not None:
                    assert om_ii is not None and om_tt is not None and om_it is not None
                    l_nbr = rch_v3_neighbor_nce_loss(
                        im=im_out,
                        s=txt_out,
                        z_ii=z_ii,
                        z_tt=z_tt,
                        z_it=z_it,
                        omega_ii=om_ii,
                        omega_tt=om_tt,
                        omega_it=om_it,
                        top_k=int(args.rch_v3_top_k),
                        z_min=float(args.rch_v3_z_min),
                        fn_z=float(args.rch_v3_fn_z),
                        tau_g=float(args.rch_v3_tau_g),
                    )
                    loss = loss + lambda_graph * l_nbr
            elif is_v2:
                assert nacg_state is not None
                r_b = nacg_state.r.index_select(0, idx_t).to(device)
                v_b = nacg_state.valid_mask.index_select(0, idx_t).to(device)
                w_b = compute_rch_sample_weights(
                    r_b,
                    gamma_r=float(args.rch_gamma_r),
                    delta=float(args.rch_delta),
                )
                out_l, out_ab = rch_nce(
                    im_out,
                    txt_out,
                    idx_t,
                    epoch=epoch - args.warmup_epoch,
                    r_batch=r_b if use_v2_rel else None,
                    gamma_r=float(args.rch_gamma_r),
                )
                if use_v2_rel:
                    l_loss = weighted_nce_softmax_loss(out_l, w_b, normalize="sum_w")
                    ab_loss = weighted_nce_softmax_loss(out_ab, w_b, normalize="sum_w")
                    lc_val = l_loss + ab_loss
                else:
                    l_loss = criterion_contrast(out_l)
                    ab_loss = criterion_contrast(out_ab)
                    lc_val = l_loss + ab_loss

                if use_v2_graph:
                    g_ii, g_tt, g_it, om_ii, om_tt, om_it = compute_centered_clean_graph(
                        imgs,
                        txts,
                        r_b,
                        v_b,
                        nacg_state,
                    )
                    lr_val = rch_contrastive_loss(
                        im=im_out,
                        s=txt_out,
                        w_batch=w_b,
                        margin=float(args.margin),
                        shift=float(args.shift),
                        false_neg_sim=g_it,
                        false_neg_conf=om_it,
                        pos_quantile=float(args.rch_pos_quantile),
                    )
                    l_graph = rch_clean_graph_loss(
                        im=im_out,
                        s=txt_out,
                        g_ii=g_ii,
                        g_tt=g_tt,
                        g_it=g_it,
                        omega_ii=om_ii,
                        omega_tt=om_tt,
                        omega_it=om_it,
                        pos_quantile=float(args.rch_pos_quantile),
                        stretch=float(args.rch_stretch),
                    )
                    loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha) + lambda_graph * l_graph
                else:
                    lr_val = rch_contrastive_loss(
                        im=im_out,
                        s=txt_out,
                        w_batch=w_b,
                        margin=float(args.margin),
                        shift=float(args.shift),
                        false_neg_sim=None,
                    )
                    loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha)
            else:
                assert nacg_state is not None
                r_b = nacg_state.r.index_select(0, idx_t).to(device)
                nbr_idx_i_b = nacg_state.nbr_idx_I.index_select(0, idx_t).to(device)
                nbr_w_i_b = nacg_state.nbr_w_I.index_select(0, idx_t).to(device)
                nbr_idx_t_b = nacg_state.nbr_idx_T.index_select(0, idx_t).to(device)
                nbr_w_t_b = nacg_state.nbr_w_T.index_select(0, idx_t).to(device)

                if use_nacg_this_ep:
                    assert anc_hx is not None and anc_hy is not None
                    hx_bridge = gather_bridged_features(anc_hx, nbr_idx_t_b, nbr_w_t_b)
                    hy_bridge = gather_bridged_features(anc_hy, nbr_idx_i_b, nbr_w_i_b)
                    out_l, out_ab = nacg_nce(
                        im_out,
                        txt_out,
                        idx_t,
                        epoch=epoch - args.warmup_epoch,
                        r_batch=r_b,
                        hx_bridge=hx_bridge,
                        hy_bridge=hy_bridge,
                        anchor_idx=nacg_state.anchor_idx,
                        nbr_idx_t=nbr_idx_t_b,
                        nbr_w_t=nbr_w_t_b,
                    )
                else:
                    hx_bridge = im_out.detach()
                    hy_bridge = txt_out.detach()
                    out_l, out_ab = contrast(im_out, txt_out, idx_t, epoch=epoch - args.warmup_epoch)

                l_loss = criterion_contrast(out_l)
                ab_loss = criterion_contrast(out_ab)
                lc_val = l_loss + ab_loss

                sim_it = None
                if use_graph_this_ep:
                    assert anc_norm_img is not None and anc_norm_txt is not None
                    x_feat_bridge = gather_bridged_features(anc_norm_img, nbr_idx_t_b, nbr_w_t_b)
                    y_feat_bridge = gather_bridged_features(anc_norm_txt, nbr_idx_i_b, nbr_w_i_b)
                    sim_it = compute_clean_cross_sim(imgs, txts, r_b, x_feat_bridge, y_feat_bridge)

                if use_nacg_this_ep or sim_it is not None:
                    lr_val = nacg_ucch_contrastive_loss(
                        im=im_out,
                        s=txt_out,
                        r_batch=r_b,
                        hx_bridge=hx_bridge,
                        hy_bridge=hy_bridge,
                        margin=float(args.margin),
                        shift=float(args.shift),
                        false_neg_sim=sim_it,
                    )
                else:
                    lr_val = criterion(im_out, txt_out)

                loss = lc_val * args.alpha + lr_val * (1.0 - args.alpha)
                if use_graph_this_ep and sim_it is not None:
                    loss = loss + lambda_graph * nacg_positive_graph_loss(im_out, txt_out, sim_it)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        lr_schedu.step(epoch)
        if not losses or not np.isfinite(losses).all():
            raise RuntimeError(f"UCCH/{args.variant} epoch {epoch} 出现无效损失")

        # 与 i1_train.py 中 retrieval_loader (_BaseDataLoaderIter.__init__) 对齐：每轮验证前消耗 1 个 int64 随机种子抽样
        torch.empty((), dtype=torch.int64).random_()
        # 验证集评估（切片方式与上游一致：先按完整 eval_batch_size 编码再切前 2000 条）
        val_imgs = _encode_codes(image_model, val_images_t, args.eval_batch_size, device, limit=eval_encode_limit)[:val_size]
        val_txts = _encode_codes(text_model, val_texts_t, args.eval_batch_size, device, limit=eval_encode_limit)[:val_size]
        val_res = evaluate_codes(
            val_imgs,
            val_txts,
            val_labels,
            val_imgs,
            val_txts,
            val_labels,
            query_chunk_size=args.map_query_chunk_size,
        )

        avg = float(val_res["avg_official"])
        if avg > best_acc:
            best_acc = avg
            best_epoch = epoch
            best_img_state = {k: v.detach().cpu().clone() for k, v in image_model.state_dict().items()}
            best_txt_state = {k: v.detach().cpu().clone() for k, v in text_model.state_dict().items()}

        ep_rec: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)),
            "val_i2t_official": float(val_res["i2t_official"]),
            "val_t2i_official": float(val_res["t2i_official"]),
            "val_avg_official": avg,
            "val_i2t_stable": float(val_res["i2t_stable"]),
            "val_t2i_stable": float(val_res["t2i_stable"]),
            "val_avg_stable": float(val_res["avg_stable"]),
            "n_query": int(val_res["n_query"]),
            "n_db": int(val_res["n_db"]),
        }

        if nacg_state is not None:
            if args.nacg_force_one or nacg_state.hat_eta_0 <= 0.0:
                empty_t = torch.empty((0, int(args.bit)), device=device)
                _, nacg_diag = update_nacg_epoch_reliability(
                    nacg_state,
                    empty_t,
                    empty_t,
                    epoch=epoch,
                    noise_mask=noise.mask,
                    force_one=args.nacg_force_one,
                    online_dominant=(is_v3 or is_v4),
                    self_consistent_storey=is_v4,
                    v5_mode=(is_v5 or is_v6),
                    refresh_bridges=refresh_bridges_v6,
                )
            else:
                hx_full = _encode_codes(image_model, train_images_t, args.eval_batch_size, device, normalize_continuous=True)
                hy_full = _encode_codes(text_model, train_texts_t, args.eval_batch_size, device, normalize_continuous=True)
                _, nacg_diag = update_nacg_epoch_reliability(
                    nacg_state,
                    hx_full,
                    hy_full,
                    epoch=epoch,
                    noise_mask=noise.mask,
                    force_one=False,
                    online_dominant=(is_v3 or is_v4),
                    self_consistent_storey=is_v4,
                    v5_mode=(is_v5 or is_v6),
                    refresh_bridges=refresh_bridges_v6,
                )
            ep_rec["pi"] = nacg_diag
            ep_rec["pi_diagnostics"] = nacg_diag

        ep_rec["epoch_seconds"] = time.perf_counter() - ep_t0
        epoch_records.append(ep_rec)

    if not args.no_save_ckpt:
        ckpt_tmp = out_dir / "best_checkpoint.pt.tmp"
        torch.save(
            {
                "best_val_epoch": int(best_epoch),
                "image_model": best_img_state,
                "text_model": best_txt_state,
            },
            ckpt_tmp,
        )
        os.replace(ckpt_tmp, out_dir / "best_checkpoint.pt")

    # 加载验证集最佳权重并评估全量干净检索库与测试查询集
    image_model.load_state_dict(best_img_state)
    text_model.load_state_dict(best_txt_state)

    if use_gpu_resident:
        clean_texts_t = torch.as_tensor(data.texts, device=device, dtype=torch.float32).contiguous()
        ret_imgs = _encode_codes(image_model, train_images_t, args.eval_batch_size, device)
        ret_txts = _encode_codes(text_model, clean_texts_t, args.eval_batch_size, device)
        q_imgs = _encode_codes(image_model, torch.as_tensor(data.query_images, device=device, dtype=torch.float32), args.eval_batch_size, device)
        q_txts = _encode_codes(text_model, torch.as_tensor(data.query_texts, device=device, dtype=torch.float32), args.eval_batch_size, device)
    else:
        ret_imgs = _encode_codes(image_model, data.images, args.eval_batch_size, device)
        ret_txts = _encode_codes(text_model, data.texts, args.eval_batch_size, device)
        q_imgs = _encode_codes(image_model, data.query_images, args.eval_batch_size, device)
        q_txts = _encode_codes(text_model, data.query_texts, args.eval_batch_size, device)

    test = evaluate_codes(
        ret_imgs,
        ret_txts,
        data.labels,
        q_imgs,
        q_txts,
        data.query_labels,
        query_chunk_size=args.map_query_chunk_size,
    )
    if args.dump_codes:
        np.savez_compressed(
            out_dir / "test_codes.npz",
            retrieval_imgs=ret_imgs,
            retrieval_txts=ret_txts,
            retrieval_labs=data.labels,
            query_imgs=q_imgs,
            query_txts=q_txts,
            query_labs=data.query_labels,
        )

    summary: Dict[str, Any] = {
        "data_name": str(args.data_name),
        "bit": int(args.bit),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "variant": str(args.variant),
        "run_key": f"{args.data_name}_n{args.noise_rate:g}_b{args.bit}_s{args.seed}_{args.variant}",
        "noise_rate": float(args.noise_rate),
        "noise_seed": int(args.noise_seed),
        "n_noisy": int(noise.mask.sum()),
        "n_train": n_data,
        "label_inconsistency": noise.label_diagnostic,
        "parsed_args": {
            "data_name": str(args.data_name),
            "data_root": str(args.data_root),
            "variant": str(args.variant),
            "seed": int(args.seed),
            "noise_rate": float(args.noise_rate),
            "noise_seed": int(args.noise_seed),
            "bit": int(args.bit),
            "alpha": float(args.alpha),
            "num_hiden_layers": [int(x) for x in args.num_hiden_layers],
            "margin": float(args.margin),
            "max_epochs": int(args.max_epochs),
            "train_batch_size": int(args.train_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "shift": float(args.shift),
            "lr": float(args.lr),
            "wd": float(args.wd),
            "optimizer": str(args.optimizer),
            "ls": str(args.ls),
            "warmup_epoch": int(args.warmup_epoch),
            "K": int(args.K),
            "T": float(args.T),
            "momentum": float(args.momentum),
            "num_workers": int(args.num_workers),
            "map_query_chunk_size": int(args.map_query_chunk_size),
            "device": str(args.device),
            "deterministic": bool(args.deterministic),
            "require_cuda": bool(args.require_cuda),
            "no_save_ckpt": bool(args.no_save_ckpt),
            "dump_codes": bool(args.dump_codes),
            "lambda_graph": lambda_graph,
            "nacg_anchors": int(args.nacg_anchors),
            "nacg_top_k": int(args.nacg_top_k),
            "nacg_tau_bridge": float(args.nacg_tau_bridge),
            "nacg_tau_null": float(args.nacg_tau_null),
            "nacg_force_one": bool(args.nacg_force_one),
            "rch_gamma_r": float(args.rch_gamma_r),
            "rch_delta": float(args.rch_delta),
            "rch_pos_quantile": float(args.rch_pos_quantile),
            "rch_stretch": float(args.rch_stretch),
            "rch_v3_delta": float(args.rch_v3_delta),
            "rch_v3_top_k": int(args.rch_v3_top_k),
            "rch_v3_z_min": float(args.rch_v3_z_min),
            "rch_v3_z_fn_thr": float(args.rch_v3_z_fn_thr),
            "rch_v3_fn_z": float(args.rch_v3_fn_z),
            "rch_v3_tau_g": float(args.rch_v3_tau_g),
            "rch_v4_twin_start_epoch": int(args.rch_v4_twin_start_epoch),
            "rch_v4_z_twin_thr": float(args.rch_v4_z_twin_thr),
            "rch_v4_z_nce_fn_thr": float(args.rch_v4_z_nce_fn_thr),
            "rch_v4_z_fn_thr": float(args.rch_v4_z_fn_thr),
            "rch_v4_z_min": float(args.rch_v4_z_min),
            "rch_v4_target_base": float(args.rch_v4_target_base),
            "rch_v4_target_slope": float(args.rch_v4_target_slope),
            "rch_v4_target_max": float(args.rch_v4_target_max),
            "rch_v4_quant_mix": float(args.rch_v4_quant_mix),
            "rch_v5_twin_start_epoch": int(args.rch_v5_twin_start_epoch),
            "rch_v5_z_twin_thr": float(args.rch_v5_z_twin_thr),
            "rch_v5_sym_delta": float(args.rch_v5_sym_delta),
            "rch_v5_fn_thr": float(args.rch_v5_fn_thr),
            "rch_v5_stretch": float(args.rch_v5_stretch),
            "rch_v5_ho_weight": float(args.rch_v5_ho_weight),
            "rch_v5_upper_weight": float(args.rch_v5_upper_weight),
            "rch_v6_sym_delta": float(args.rch_v6_sym_delta),
            "rch_v6_fn_thr": float(args.rch_v6_fn_thr),
            "rch_v6_ho_weight": float(args.rch_v6_ho_weight),
            "rch_v6_bridge_sharpen": float(args.rch_v6_bridge_sharpen),
            "rch_v6_use_fn_adjust": bool(
                is_v6 and (args.variant not in ("RCH-v6-nog", "RCH-v6-gonly")) and (float(lambda_graph) > 0.0 or args.variant == "RCH-v6-fnonly")
            ),
        },
        "best_val_epoch": int(best_epoch),
        "best_val_avg_official": float(best_acc),
        "test_i2t_official": float(test["i2t_official"]),
        "test_t2i_official": float(test["t2i_official"]),
        "test_avg_official": float(test["avg_official"]),
        "test_i2t_stable": float(test["i2t_stable"]),
        "test_t2i_stable": float(test["t2i_stable"]),
        "test_avg_stable": float(test["avg_stable"]),
        "n_query": int(test["n_query"]),
        "n_db": int(test["n_db"]),
        "epoch_records": epoch_records,
        "eval_records": test["eval_records"],
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(device.type == "cuda"),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    if nacg_state is not None:
        summary["nacg_diagnostics"] = {
            "hat_eta_0": float(nacg_state.hat_eta_0),
            "hat_eta_raw_0": float(nacg_state.hat_eta_raw_0),
            "cold_start_auc": nacg_state.cold_start_auc,
            "zero_noise_exact_gate": bool(nacg_state.hat_eta_0 <= 0.0),
            "self_consistent_storey": bool(nacg_state.self_consistent_storey),
            "v5_mode": bool(is_v5 or is_v6),
            "refresh_bridges": bool(refresh_bridges_v6),
            "last_epoch_pi": epoch_records[-1].get("pi") if epoch_records else None,
        }
    write_json_atomic(out_dir / "summary.json", summary)
    print(
        "SUMMARY " + json.dumps({k: v for k, v in summary.items() if k not in ("epoch_records", "eval_records")}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
