"""I2 前置诊断（多次掩码均值—方差解耦）与 I1 逐位可靠记忆库核心算法模块（baseline/ours/i1.py）。

严格遵循 `.sci-skills/design/M3-REVISE-001/method-revision.md` 与 `.sci-skills/design/M3-REVISE-002/method-revision.md` 规范：
  1. 共享的 K 次掩码视图（默认 K=4，rho_bar=0.8，mask_rescale=False）：
     完整表示 h_x = f_x(x_i), h_y = f_y(y_i) 仅计算一次；
     K 个独立 Bernoulli 掩码视图 (h_x^k, h_y^k) 由 I2 与 I1 共享，不重复做两套 K 次前向。
  2. I2 多次掩码均值与方差解耦：
     - c_i^k = 0.5 * (<h_x, h_y^k> + <h_x^k, h_y>)
     - c_bar_i = (1/K) * sum_k c_i^k
     - s_i = sqrt((1/K) * sum_k (c_i^k - c_bar_i)^2)  （总体标准差，ddof=0）
     - 单掩码可信度：pi_single_i = GMM(c_i^1)
     - 均值版可信度：pi_mean_i = p_i = GMM(c_bar_i)
     - 均值+方差可信度：pi_var_i = clip(p_i * exp(-lambda * s_i), 0, 1)
       （若 GMM 触发恒定信号回退，方差乘子固定取 1，不在无法拟合后验时额外降权）
  3. 先导探针（probe，总视图数 K=4）：
     - 复用原 MCR-A 第一视图随机流计算 c^1，由独立探针随机流补充 c^2..c^4；
     - 新增视图仅输出诊断，不改变训练样本权重或记忆库，也不消耗原诊断掩码随机流；
     - 内置 c^1 与原单掩码路径的逐位一致性核验。
  4. I1 逐位跨模态符号一致性与逐位动量记忆库：
     - sgn 直接沿用 torch.sign（零值保持为 0，符号相等按数值相等 == 计算）：
       q[i, l] = (1 / (2K)) * sum_k (1[sgn(h_x^k[i, l]) == sgn(h_y[i, l])] + 1[sgn(h_y^k[i, l]) == sgn(h_x[i, l])])
     - 合并向量 v = (l + ab) / 2，v <- v / ||v||_2
     - 逐位有效动量 m[i, l] = 1 - (1 - m) * pi_i * q[i, l]
     - 逐位更新 B_new[i, l] = m[i, l] * B_old[i, l] + (1 - m[i, l]) * v[i, l]，随后对整向量做 L2 归一化
     - 出现零范数时保留旧槽位 B_old[i] 并记录零范数数值事件，绝不静默产生 NaN。
  5. RAND-Q 等写入预算随机位对照：
     - 对每个样本独立随机置换其 q 向量的位顺序 q_perm[i, l] = q[i, sigma_i(l)]；
     - 逐样本精确保持位可靠性分布与 mean_l(q_perm[i, l]) == mean_l(q[i, l])，共享相同 pi 来源。
  6. Soft-Q 与 SOFT-RAND-Q（M3-REVISE-002）：
     - g[i,l]=(1-beta)+beta*q[i,l]/mean(q[i,:])，零均值样本回退 g=1，固定 beta=0.5；
     - 每样本 mean(g)=1，平均写入系数与 A-K4 相同；0<beta<1 时无 q=0 永久冻结；
     - SOFT-RAND-Q 先置换 q 再计算 g，保持每样本 g 分布与均值，检验位身份作用；
     - 可选记录与同输入标量更新的方向余弦，不改变记忆更新结果。
  7. 双路径机制消融：
     - L-pi 仅用 pi_single 加权 Lc/Lr，记忆更新保持 UCCH 标量动量；
     - M-pi 仅用 pi_single 门控记忆写入，Lc/Lr 保持原始无权重损失；
     - PERM-pi 保留双门控，每轮独立置换训练缓冲中的 pi_single；
       保持均值与分布、隔离随机流、预热不置换，并保留原始缓冲供下一轮使用。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-not-found]

from ours.mcr import (  # type: ignore[import-not-found]
    apply_feature_mask,
    build_correspondence_permutation,
    compute_consistency_signal,
    compute_per_sample_lr_signal,
    compute_sample_weights,
    fit_gmm_reliability,
    inject_correspondence_noise,
    make_derangement,
    weighted_contrastive_loss,
    weighted_nce_softmax_loss,
)

I1Variant = Literal[
    "off",
    "probe",
    "A-K1",
    "A-K1-G",
    "A-K4",
    "I1",
    "RAND-Q",
    "SOFT-Q",
    "SOFT-RAND-Q",
    "A-K4-V",
    "I1-V",
    "L-pi",
    "M-pi",
    "PERM-pi",
]

VALID_I1_VARIANTS: Tuple[str, ...] = (
    "off",
    "probe",
    "A-K1",
    "A-K1-G",
    "A-K4",
    "I1",
    "RAND-Q",
    "SOFT-Q",
    "SOFT-RAND-Q",
    "A-K4-V",
    "I1-V",
    "L-pi",
    "M-pi",
    "PERM-pi",
)

DEFAULT_I2_LAMBDA_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
DEFAULT_SOFT_BETA: float = 0.5

__all__ = [
    "DEFAULT_I2_LAMBDA_GRID",
    "DEFAULT_SOFT_BETA",
    "I1NCEAverage",
    "I1Variant",
    "VALID_I1_VARIANTS",
    "apply_feature_mask",
    "build_correspondence_permutation",
    "compute_bitwise_momentum",
    "compute_bitwise_sign_consistency",
    "compute_consistency_signal",
    "compute_i2_reliabilities",
    "compute_multiview_i2_signals",
    "compute_per_sample_lr_signal",
    "compute_pi_var_from_mean",
    "compute_sample_weights",
    "compute_soft_bit_gate",
    "fit_gmm_reliability",
    "fit_gmm_reliability_flagged",
    "inject_correspondence_noise",
    "make_derangement",
    "normalize_variant_name",
    "permute_pi_single",
    "permute_q_per_sample",
    "prepare_applied_pi",
    "resolve_dual_path_flags",
    "summarize_i1_bitwise_diagnostics",
    "summarize_i2_diagnostics",
    "update_memory_bitwise",
    "weighted_contrastive_loss",
    "weighted_nce_softmax_loss",
]


def normalize_variant_name(
    variant: str,
    use_var: bool = False,
    mcr_a_flag: bool = False,
) -> I1Variant:
    """规范化变体标识并处理默认关闭的方差分支开关。

    支持显式传入 'off', 'probe', 'A-K1', 'A-K4', 'I1', 'RAND-Q', 'SOFT-Q', 'SOFT-RAND-Q', 'A-K4-V', 'I1-V', 'L-pi', 'M-pi', 'PERM-pi'（及别名 'A' -> 'A-K1'）。
    当 use_var=True 时，'A-K4' 升级为 'A-K4-V'，'I1' 升级为 'I1-V'。
    """
    raw = str(variant).strip()
    if not raw or raw == "off":
        if mcr_a_flag:
            return "A-K1"
        return "off"
    if raw == "A":
        raw = "A-K1"
    if raw not in VALID_I1_VARIANTS:
        raise ValueError(f"不支持的变体名称 variant={variant!r}，可选值: {VALID_I1_VARIANTS}")

    if use_var:
        if raw == "A-K4":
            return "A-K4-V"
        if raw == "I1":
            return "I1-V"

    return raw  # type: ignore[return-value]


def resolve_dual_path_flags(variant: I1Variant) -> Tuple[bool, bool, bool]:
    """返回 (loss_weighting, memory_gating, perm_pi) 三个布尔标志。

    四组合语义：
      - off:     (False, False, False) — 纯 UCCH，不生成也不使用 pi；
      - L-pi:    (True,  False, False) — 仅损失可信度加权，记忆库退化为上游标量动量；
      - M-pi:    (False, True,  False) — 仅记忆库标量门控，损失保持未加权；
      - A-K1:    (True,  True,  False) — 现有主线行为；
      - PERM-pi: (True,  True,  True)  — 两路均开，但每轮对训练缓冲 pi 做一次逐样本置换。
    既有变体（probe/A-K4/I1/RAND-Q/SOFT-Q/SOFT-RAND-Q/A-K4-V/I1-V）保持原有行为不变。
    """
    loss_weighting = variant in (
        "A-K1",
        "A-K1-G",
        "L-pi",
        "PERM-pi",
        "probe",
        "A-K4",
        "A-K4-V",
        "I1",
        "I1-V",
        "RAND-Q",
        "SOFT-Q",
        "SOFT-RAND-Q",
    )
    memory_gating = variant in (
        "A-K1",
        "A-K1-G",
        "M-pi",
        "PERM-pi",
        "probe",
        "A-K4",
        "A-K4-V",
        "I1",
        "I1-V",
        "RAND-Q",
        "SOFT-Q",
        "SOFT-RAND-Q",
    )
    perm_pi = variant == "PERM-pi"
    return bool(loss_weighting), bool(memory_gating), bool(perm_pi)


def permute_pi_single(
    pi: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """对一维权值 pi 做一次全局随机置换，严格保持均值、排序后值和直方图不变。

    使用独立的 torch.Generator，不污染训练、掩码、probe 或 RAND-Q 随机流。
    返回 detach 后的置换结果，置换前后 sorted(pi) 逐元素相等。
    """
    if pi.ndim != 1:
        raise ValueError(f"pi 必须是 1D 张量，实际为 {pi.ndim}D")
    n = int(pi.numel())
    gen_device = generator.device
    with torch.no_grad():
        rand_scores = torch.rand(n, dtype=torch.float64, device=gen_device, generator=generator)
        perm = torch.argsort(rand_scores).to(device=pi.device, dtype=torch.long)
        pi_perm = pi.detach()[perm]
    return pi_perm.detach()


def prepare_applied_pi(
    pi: torch.Tensor,
    perm_enabled: bool,
    warmup: bool,
    generator: torch.Generator,
    permutation_seed: int,
    perm_seed_offset: int,
    source_snapshot_from_epoch: Optional[int],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """构造本轮实际应用的 pi 快照及 PERM-pi 守恒诊断。

    非 PERM 变体保持原值，但返回独立快照以避免后续缓冲 copy_ 覆盖。
    PERM-pi 在 warmup 轮不调用随机数；非 warmup 轮仅调用一次
    ``permute_pi_single``，因此同轮所有 batch 可共享本函数返回的同一快照。
    原始 ``pi`` 不被修改。
    """
    if not perm_enabled:
        return pi.detach().clone(), {"enabled": False}

    rng_state_before = generator.get_state().clone()
    if warmup:
        applied = pi.detach()
        permutation_applied = False
    else:
        applied = permute_pi_single(pi, generator)
        permutation_applied = True
    rng_state_after = generator.get_state().clone()

    before_np = pi.detach().cpu().numpy().astype(np.float64, copy=True)
    after_np = applied.detach().cpu().numpy().astype(np.float64, copy=True)
    before_sorted = np.sort(before_np)
    after_sorted = np.sort(after_np)
    before_hist, _ = np.histogram(before_np, bins=20, range=(0.0, 1.0))
    after_hist, _ = np.histogram(after_np, bins=20, range=(0.0, 1.0))
    mean_abs_diff = float(abs(float(before_np.mean()) - float(after_np.mean())))
    sorted_max_abs_diff = float(np.max(np.abs(before_sorted - after_sorted))) if before_np.size else 0.0
    histogram_max_abs_diff = int(np.max(np.abs(before_hist.astype(np.int64) - after_hist.astype(np.int64)))) if before_np.size else 0
    rng_unchanged = bool(torch.equal(rng_state_before, rng_state_after))
    diagnostics = {
        "enabled": True,
        "applied": bool(permutation_applied),
        "warmup": bool(warmup),
        "source_snapshot_from_epoch": source_snapshot_from_epoch,
        "permutation_seed": int(permutation_seed),
        "perm_seed_offset": int(perm_seed_offset),
        "sample_count": int(before_np.size),
        "mean_before": float(before_np.mean()) if before_np.size else None,
        "mean_after": float(after_np.mean()) if after_np.size else None,
        "mean_abs_diff": mean_abs_diff,
        "sorted_max_abs_diff": sorted_max_abs_diff,
        "histogram_max_abs_diff": histogram_max_abs_diff,
        "mean_preserved": bool(mean_abs_diff <= 1e-12),
        "sorted_values_preserved": bool(np.array_equal(before_sorted, after_sorted)),
        "histogram_preserved": bool(np.array_equal(before_hist, after_hist)),
        "rng_state_unchanged": rng_unchanged,
        "warmup_rng_preserved": rng_unchanged if warmup else None,
        "random_stream": "independent_perm_gen",
    }
    return applied.detach(), diagnostics


def _stack_views(
    views: Union[Sequence[torch.Tensor], torch.Tensor],
    expected_batch: int,
    expected_bit: int,
) -> torch.Tensor:
    """将掩码视图序列或张量统一整理为形状 (K, B, bit) 的张量。"""
    views_any: Any = views
    if isinstance(views_any, torch.Tensor):
        if views_any.ndim == 2:
            stacked = views_any.unsqueeze(0)
        elif views_any.ndim == 3:
            if views_any.shape[1] == expected_batch and views_any.shape[2] == expected_bit:
                stacked = views_any
            elif views_any.shape[0] == expected_batch and views_any.shape[2] == expected_bit:
                stacked = views_any.permute(1, 0, 2).contiguous()
            else:
                raise ValueError(
                    f"3D 视图张量形状 {tuple(views_any.shape)} 与 (K, B={expected_batch}, bit={expected_bit}) 不匹配"
                )
        else:
            raise ValueError(f"视图张量维度必须为 2 或 3，实际为 {views_any.ndim}")
    else:
        view_list = list(views_any)
        if not view_list:
            raise ValueError("掩码视图列表不能为空（要求 K >= 1）")
        stacked = torch.stack(view_list, dim=0)

    if stacked.shape[0] < 1:
        raise ValueError(f"掩码视图数量 K 必须 >= 1，当前 K={stacked.shape[0]}")
    if stacked.shape[1] != expected_batch or stacked.shape[2] != expected_bit:
        raise ValueError(
            f"掩码视图形状 {tuple(stacked.shape)} 与预期 (K, B={expected_batch}, bit={expected_bit}) 不匹配"
        )
    return stacked


def compute_bitwise_sign_consistency(
    h_x: torch.Tensor,
    h_y: torch.Tensor,
    h_x_views: Union[Sequence[torch.Tensor], torch.Tensor],
    h_y_views: Union[Sequence[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """计算 2K 个跨模态掩码—完整视图位符号一致比例 q[i, l] in [0, 1]。

    严格遵循 M3 第 5.1 节定义：
      sgn 直接采用 torch.sign（零值保持为 0，符号相等按数值相等 == 计算）：
      q[i, l] = (1 / (2K)) * sum_{k=1}^K (
          1[sgn(h_x^k[i, l]) == sgn(h_y[i, l])] + 1[sgn(h_y^k[i, l]) == sgn(h_x[i, l])]
      )
    全程 detach，不向掩码视图或主干网络反传梯度。
    """
    if h_x.ndim != 2 or h_y.ndim != 2 or h_x.shape != h_y.shape:
        raise ValueError(
            f"h_x 与 h_y 必须是同形状二维张量 (B, bit)，实际为 {tuple(h_x.shape)} 与 {tuple(h_y.shape)}"
        )
    batch_size, bit = int(h_x.shape[0]), int(h_x.shape[1])
    hx_stack = _stack_views(h_x_views, batch_size, bit)
    hy_stack = _stack_views(h_y_views, batch_size, bit)
    if hx_stack.shape[0] != hy_stack.shape[0]:
        raise ValueError(
            f"图像与文本掩码视图数 K 不一致: {hx_stack.shape[0]} vs {hy_stack.shape[0]}"
        )
    k_views = int(hx_stack.shape[0])

    with torch.no_grad():
        sgn_x = torch.sign(h_x.detach()).unsqueeze(0)  # (1, B, bit)
        sgn_y = torch.sign(h_y.detach()).unsqueeze(0)  # (1, B, bit)
        sgn_xk = torch.sign(hx_stack.detach())  # (K, B, bit)
        sgn_yk = torch.sign(hy_stack.detach())  # (K, B, bit)

        match_xk_y = (sgn_xk == sgn_y).to(dtype=torch.float32)
        match_yk_x = (sgn_yk == sgn_x).to(dtype=torch.float32)
        q = (match_xk_y.sum(dim=0) + match_yk_x.sum(dim=0)) / float(2 * k_views)

    return q.detach()


def compute_multiview_i2_signals(
    h_x: torch.Tensor,
    h_y: torch.Tensor,
    h_x_views: Union[Sequence[torch.Tensor], torch.Tensor],
    h_y_views: Union[Sequence[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """基于共享的完整表示与 K 个掩码视图，同时计算 I2 信号 (c_views, c_bar, s_std) 与 I1 位一致性 q。

    返回：
      - c_views: 形状 (B, K)，每个视图的跨模态内积一致性 c_i^k
      - c_bar: 形状 (B,)，K 个视图的均值 c_bar_i = (1/K) * sum_k c_i^k
      - s_std: 形状 (B,)，K 个视图的总体标准差 s_i = sqrt((1/K) * sum_k (c_i^k - c_bar_i)^2) (ddof=0)
      - q: 形状 (B, bit)，2K 个跨模态位符号一致比例 q[i, l]
    """
    if h_x.ndim != 2 or h_y.ndim != 2 or h_x.shape != h_y.shape:
        raise ValueError(
            f"h_x 与 h_y 必须是同形状二维张量 (B, bit)，实际为 {tuple(h_x.shape)} 与 {tuple(h_y.shape)}"
        )
    batch_size, bit = int(h_x.shape[0]), int(h_x.shape[1])
    hx_stack = _stack_views(h_x_views, batch_size, bit)
    hy_stack = _stack_views(h_y_views, batch_size, bit)
    if hx_stack.shape[0] != hy_stack.shape[0]:
        raise ValueError(
            f"图像与文本掩码视图数 K 不一致: {hx_stack.shape[0]} vs {hy_stack.shape[0]}"
        )
    k_views = int(hx_stack.shape[0])

    with torch.no_grad():
        hx_det = h_x.detach()
        hy_det = h_y.detach()
        c_list: List[torch.Tensor] = []
        for k in range(k_views):
            c_k = compute_consistency_signal(
                hx_det,
                hy_det,
                hx_stack[k].detach(),
                hy_stack[k].detach(),
            )
            c_list.append(c_k.to(dtype=torch.float32))

        c_views = torch.stack(c_list, dim=1)  # (B, K)
        c_bar = c_views.mean(dim=1)  # (B,)
        if k_views == 1:
            s_std = torch.zeros_like(c_bar)
        else:
            diff = c_views - c_bar.unsqueeze(1)
            s_std = torch.sqrt(torch.mean(diff * diff, dim=1).clamp(min=0.0))

        q = compute_bitwise_sign_consistency(hx_det, hy_det, hx_stack, hy_stack)

    return c_views.detach(), c_bar.detach(), s_std.detach(), q.detach()


def fit_gmm_reliability_flagged(
    signals: np.ndarray,
    mode: Literal["consistency", "loss"] = "consistency",
    random_state: int = 1123,
) -> Tuple[np.ndarray, float, float, bool]:
    """对一维信号拟合双分量 GMM 并额外返回恒定信号回退标记 fallback。"""
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    s_min = float(np.min(sig))
    s_max = float(np.max(sig))
    fallback = bool((s_max - s_min) < 1e-12)
    pi, clean_mean, noisy_mean = fit_gmm_reliability(
        sig,
        mode=mode,
        random_state=random_state,
    )
    return pi, clean_mean, noisy_mean, fallback


def compute_pi_var_from_mean(
    pi_mean: Union[np.ndarray, torch.Tensor],
    s_std: Union[np.ndarray, torch.Tensor],
    lambda_var: float = 1.0,
    fallback: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """根据 M3 第 4.1 节公式计算 pi_var_i = clip(p_i * exp(-lambda * s_i), 0, 1)。

    若 GMM 触发恒定信号回退（fallback=True），方差乘子取 1，不在无法拟合后验时额外降权。
    """
    pi_any: Any = pi_mean
    s_any: Any = s_std
    if isinstance(pi_any, torch.Tensor):
        p_det = pi_any.detach().to(dtype=torch.float32).clamp(min=0.0, max=1.0)
        if fallback or float(lambda_var) == 0.0:
            return p_det.clone()
        s_det = (
            s_any.detach().to(device=p_det.device, dtype=torch.float32)
            if isinstance(s_any, torch.Tensor)
            else torch.as_tensor(s_any, dtype=torch.float32, device=p_det.device)
        ).clamp(min=0.0)
        factor = torch.exp(-float(lambda_var) * s_det.to(dtype=torch.float64)).to(dtype=torch.float32)
        return (p_det * factor).clamp(min=0.0, max=1.0).detach()

    p_np = np.clip(np.asarray(pi_any, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if fallback or float(lambda_var) == 0.0:
        return p_np.astype(np.float32)
    s_np = np.maximum(np.asarray(s_any, dtype=np.float64).reshape(-1), 0.0)
    factor_np = np.exp(-float(lambda_var) * s_np)
    return np.clip(p_np * factor_np, 0.0, 1.0).astype(np.float32)


def compute_i2_reliabilities(
    c_views: Union[np.ndarray, torch.Tensor],
    s_std: Optional[Union[np.ndarray, torch.Tensor]] = None,
    lambda_var: float = 1.0,
    mode: Literal["consistency", "loss"] = "consistency",
    random_state: int = 1123,
    lambda_grid: Sequence[float] = DEFAULT_I2_LAMBDA_GRID,
    include_gmm_params: bool = False,
) -> Dict[str, Any]:
    """从 (N, K) 视图一致性矩阵计算 I2 的单掩码、均值、均值+方差可信度及等预算敏感性序列。"""
    c_any: Any = c_views
    s_any: Any = s_std
    if isinstance(c_any, torch.Tensor):
        c_np = c_any.detach().cpu().numpy().astype(np.float64)
    else:
        c_np = np.asarray(c_any, dtype=np.float64)

    if c_np.ndim == 1:
        c_np = c_np.reshape(-1, 1)
    if c_np.ndim != 2 or c_np.shape[1] < 1:
        raise ValueError(f"c_views 必须是二维数组 (N, K)，实际形状为 {c_np.shape}")

    k_views = int(c_np.shape[1])
    c_1 = c_np[:, 0].astype(np.float32)
    c_bar = np.mean(c_np, axis=1).astype(np.float32)

    if s_any is None:
        s_arr = np.std(c_np, axis=1, ddof=0).astype(np.float32)
    elif isinstance(s_any, torch.Tensor):
        s_arr = s_any.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        s_arr = np.asarray(s_any, dtype=np.float32).reshape(-1)

    gmm_params: Optional[Dict[str, Any]] = None
    if include_gmm_params:
        from ours.separation import fit_gmm_reliability_with_params  # type: ignore[import-not-found]
        fitted = fit_gmm_reliability_with_params(c_1, mode=mode, random_state=random_state)
        gmm_params = fitted
        pi_single = fitted["pi"]
        single_c_mean = fitted["clean_mean"]
        single_n_mean = fitted["noisy_mean"]
        fb_single = fitted["fallback"]
    else:
        pi_single, single_c_mean, single_n_mean, fb_single = fit_gmm_reliability_flagged(
            c_1, mode=mode, random_state=random_state,
        )
    if k_views == 1:
        pi_mean = pi_single.copy()
        mean_c_mean = single_c_mean
        mean_n_mean = single_n_mean
        fb_mean = fb_single
    else:
        pi_mean, mean_c_mean, mean_n_mean, fb_mean = fit_gmm_reliability_flagged(
            c_bar,
            mode=mode,
            random_state=random_state,
        )

    pi_var = compute_pi_var_from_mean(
        pi_mean,
        s_arr,
        lambda_var=float(lambda_var),
        fallback=fb_mean,
    )
    assert isinstance(pi_var, np.ndarray)

    pi_var_grid: Dict[str, np.ndarray] = {}
    for lam in lambda_grid:
        key = f"{float(lam):g}"
        arr = compute_pi_var_from_mean(
            pi_mean,
            s_arr,
            lambda_var=float(lam),
            fallback=fb_mean,
        )
        assert isinstance(arr, np.ndarray)
        pi_var_grid[key] = arr

    return {
        "gmm_params": gmm_params,
        "k_views": k_views,
        "c_1": c_1,
        "c_bar": c_bar,
        "s_std": s_arr,
        "pi_single": pi_single,
        "pi_mean": pi_mean,
        "pi_var": pi_var,
        "pi_var_grid": pi_var_grid,
        "lambda_var": float(lambda_var),
        "gmm_fallback_single": fb_single,
        "gmm_fallback_mean": fb_mean,
        "single_gmm_clean_mean": single_c_mean,
        "single_gmm_noisy_mean": single_n_mean,
        "mean_gmm_clean_mean": mean_c_mean,
        "mean_gmm_noisy_mean": mean_n_mean,
    }


def permute_q_per_sample(
    q: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对每个样本独立随机置换其 q 向量的位顺序：q_perm[i, l] = q[i, sigma_i(l)]。

    使用独立的 torch.Generator，不消耗全局 torch / numpy / random 随机流。
    返回 (q_perm, perm_indices)，其中每个样本的位分布与 mean_l(q[i, l]) 完全不变。
    """
    if q.ndim != 2:
        raise ValueError(f"q 必须是二维张量 (N, bit)，实际形状为 {tuple(q.shape)}")
    n_samples, bit = int(q.shape[0]), int(q.shape[1])
    gen_device = generator.device

    with torch.no_grad():
        rand_scores = torch.rand(
            (n_samples, bit),
            dtype=torch.float64,
            device=gen_device,
            generator=generator,
        )
        perm_indices = torch.argsort(rand_scores, dim=1).to(device=q.device, dtype=torch.long)
        q_perm = torch.gather(q.detach(), dim=1, index=perm_indices)

    return q_perm.detach(), perm_indices.detach()


def compute_soft_bit_gate(
    q: torch.Tensor,
    beta: float = DEFAULT_SOFT_BETA,
    eps: float = 1e-12,
) -> torch.Tensor:
    """计算样本内均值归一化软位门控，零均值样本回退为标量门控。"""
    if q.ndim != 2:
        raise ValueError(f"q 必须是二维张量 (N, bit)，实际形状为 {tuple(q.shape)}")
    beta_value = float(beta)
    if not 0.0 <= beta_value <= 1.0:
        raise ValueError(f"beta 必须位于 [0, 1]，当前为 {beta_value}")
    if not bool(torch.isfinite(q).all().item()):
        raise ValueError("q 包含 NaN 或 Inf")

    with torch.no_grad():
        out_dtype = q.dtype if q.is_floating_point() else torch.float32
        if beta_value == 0.0:
            return torch.ones_like(q, dtype=out_dtype).detach()
        q_f64 = q.detach().to(dtype=torch.float64).clamp(0.0, 1.0)
        q_mean = q_f64.mean(dim=1, keepdim=True)
        nonzero_mean = q_mean > float(eps)
        safe_mean = torch.where(nonzero_mean, q_mean, torch.ones_like(q_mean))
        gate = (1.0 - beta_value) + beta_value * q_f64 / safe_mean
        gate = torch.where(nonzero_mean, gate, torch.ones_like(gate))
        return gate.to(dtype=out_dtype).detach()


def compute_bitwise_momentum(
    momentum: float,
    pi: torch.Tensor,
    q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算逐位有效动量 m[i, l] = 1 - (1 - m) * pi_i * q[i, l] 与写入率 1 - m[i, l]。

    全程 detach 并以 float64 中间精度计算，确保当 pi=1, q=1 时与原标量动量 m 逐位一致。
    原始 q 位于 [0,1]；Soft-Q 的均值归一化门控允许大于 1，因此这里只限制非负。
    """
    pi_det = pi.detach()
    batch_size = int(pi_det.numel())
    pi_col_f64 = pi_det.view(batch_size, 1).to(dtype=torch.float64).clamp(0.0, 1.0)

    if q is None:
        gate_f64 = pi_col_f64
        out_dtype = pi_det.dtype if pi_det.is_floating_point() else torch.float32
    else:
        if q.ndim != 2 or int(q.shape[0]) != batch_size:
            raise ValueError(
                f"q 形状 {tuple(q.shape)} 必须与 batch_size={batch_size} 的二维 (B, bit) 匹配"
            )
        q_f64 = q.detach().to(device=pi_det.device, dtype=torch.float64).clamp(min=0.0)
        gate_f64 = pi_col_f64 * q_f64
        out_dtype = q.dtype if q.is_floating_point() else torch.float32

    one_minus_m_f64 = (1.0 - float(momentum)) * gate_f64
    m_f64 = 1.0 - one_minus_m_f64
    return (
        m_f64.to(dtype=out_dtype).detach(),
        one_minus_m_f64.to(dtype=out_dtype).detach(),
    )


def update_memory_bitwise(
    b_old: torch.Tensor,
    l: torch.Tensor,
    ab: torch.Tensor,
    momentum: float,
    pi: Optional[torch.Tensor] = None,
    q: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
    compute_scalar_direction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """按 M3 第 5.2 节规则在 no_grad 下执行逐位记忆库更新并处理零范数边界。

    规则：
      1. v = (l + ab) / 2，v <- v / ||v||_2
      2. m[i, l] = 1 - (1 - m) * pi_i * q[i, l]
      3. B_new[i, l] = m[i, l] * B_old[i, l] + (1 - m[i, l]) * v[i, l]
      4. B_new[i] <- B_new[i] / ||B_new[i]||_2
      5. 若合并向量 v[i] 或更新后向量 B_new[i] 的 L2 范数 <= eps（或非有限），
         该槽位保留旧值 B_old[i] 并记录零范数事件，不产生 NaN。
      6. 当无零范数事件时，采用与 MCRNCEAverage 相同的就地张量算子顺序，保证 pi=q=1 时 0-ULP 逐位一致。
    """
    if b_old.ndim != 2 or l.ndim != 2 or ab.ndim != 2:
        raise ValueError("b_old, l, ab 均必须为 (B, bit) 二维张量")
    if b_old.shape != l.shape or b_old.shape != ab.shape:
        raise ValueError(
            f"b_old {tuple(b_old.shape)}, l {tuple(l.shape)}, ab {tuple(ab.shape)} 形状不一致"
        )

    batch_size, bit = int(b_old.shape[0]), int(b_old.shape[1])
    device = b_old.device
    dtype = b_old.dtype

    with torch.no_grad():
        if pi is None:
            pi_tensor = torch.ones((batch_size, 1), dtype=dtype, device=device)
        else:
            pi_tensor = pi.detach().to(device=device, dtype=dtype).view(batch_size, 1)

        m_eff, one_minus_m = compute_bitwise_momentum(
            momentum=float(momentum),
            pi=pi_tensor,
            q=q.detach().to(device=device, dtype=dtype) if q is not None else None,
        )
        if m_eff.shape[1] == 1 and bit > 1:
            m_mat = m_eff.expand(batch_size, bit)
        else:
            m_mat = m_eff

        l_merged = (l.detach() + ab.detach()) / 2.0
        v_norm = l_merged.norm(dim=1, keepdim=True)
        v_zero_mask = (v_norm.squeeze(1) <= float(eps)) | (~torch.isfinite(v_norm.squeeze(1)))

        def scalar_direction_stats(candidate: torch.Tensor, v_unit: torch.Tensor) -> Dict[str, Any]:
            if not compute_scalar_direction:
                return {}
            scalar_write = (1.0 - float(momentum)) * pi_tensor
            scalar_m = 1.0 - scalar_write
            scalar_candidate = b_old.detach() * scalar_m + v_unit * scalar_write
            scalar_norm = scalar_candidate.norm(dim=1, keepdim=True)
            candidate_norm = candidate.norm(dim=1, keepdim=True)
            valid = (
                (scalar_norm.squeeze(1) > float(eps))
                & (candidate_norm.squeeze(1) > float(eps))
                & torch.isfinite(scalar_norm.squeeze(1))
                & torch.isfinite(candidate_norm.squeeze(1))
            )
            safe_scalar_norm = torch.where(valid.unsqueeze(1), scalar_norm, torch.ones_like(scalar_norm))
            safe_candidate_norm = torch.where(valid.unsqueeze(1), candidate_norm, torch.ones_like(candidate_norm))
            cosine = (
                (candidate / safe_candidate_norm) * (scalar_candidate / safe_scalar_norm)
            ).sum(dim=1)
            count = int(valid.sum().item())
            if count == 0:
                return {"scalar_direction_count": 0, "scalar_direction_cosine_sum": 0.0, "scalar_direction_cosine_min": None}
            valid_cosine = cosine[valid]
            return {
                "scalar_direction_count": count,
                "scalar_direction_cosine_sum": float(valid_cosine.sum().item()),
                "scalar_direction_cosine_min": float(valid_cosine.min().item()),
            }

        if not bool(torch.any(v_zero_mask).item()):
            l_merged.div_(v_norm)
            l_pos = b_old.detach().clone()
            l_pos.mul_(m_eff)
            l_pos.add_(torch.mul(l_merged, one_minus_m))
            b_norm = l_pos.norm(dim=1, keepdim=True)
            b_zero_mask = (b_norm.squeeze(1) <= float(eps)) | (~torch.isfinite(b_norm.squeeze(1)))
            if not bool(torch.any(b_zero_mask).item()):
                l_pos.div_(b_norm)
                direction_stats = scalar_direction_stats(l_pos, l_merged)
                return (
                    l_pos.detach(),
                    m_mat.detach(),
                    {
                        "zero_norm_count": 0,
                        "zero_v_norm_count": 0,
                        "zero_b_norm_count": 0,
                        "zero_norm_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
                        **direction_stats,
                    },
                )
            v_unit = l_merged
        else:
            # 存在合并向量零范数槽位：逐槽位安全归一化
            safe_v_norm = torch.where(v_zero_mask.unsqueeze(1), torch.ones_like(v_norm), v_norm)
            v_unit = torch.where(
                v_zero_mask.unsqueeze(1),
                torch.zeros_like(l_merged),
                l_merged / safe_v_norm,
            )

        # 若某样本整个写入率 (1 - m[i, :]) 全为 0（如 pi_i == 0 或 q[i, :] == 0），则无需依赖 v_unit[i]
        write_active_mask = torch.any(one_minus_m > 0.0, dim=1)
        effective_v_zero_mask = v_zero_mask & write_active_mask

        candidate = b_old.detach() * m_eff + v_unit * one_minus_m
        b_norm = candidate.norm(dim=1, keepdim=True)
        b_zero_mask = (b_norm.squeeze(1) <= float(eps)) | (~torch.isfinite(b_norm.squeeze(1)))

        zero_row_mask = effective_v_zero_mask | b_zero_mask
        safe_b_norm = torch.where(zero_row_mask.unsqueeze(1), torch.ones_like(b_norm), b_norm)
        normalized_candidate = candidate / safe_b_norm
        l_pos_safe = torch.where(
            zero_row_mask.unsqueeze(1),
            b_old.detach().clone(),
            normalized_candidate,
        )

        direction_stats = scalar_direction_stats(l_pos_safe, v_unit)
        return (
            l_pos_safe.detach(),
            m_mat.detach(),
            {
                "zero_norm_count": int(zero_row_mask.sum().item()),
                "zero_v_norm_count": int(effective_v_zero_mask.sum().item()),
                "zero_b_norm_count": int(b_zero_mask.sum().item()),
                "zero_norm_mask": zero_row_mask.detach(),
                **direction_stats,
            },
        )


class I1NCEAverage(nn.Module):
    """支持标量门控、原始/软位门控及零范数保护的记忆库包装器。

    - 当 pi=None 且 q=None 时，直接调用上游 base_nce(l, ab, y, idx=idx, epoch=epoch)，零额外操作；
    - 当 q=None 时，按样本标量 pi_i 更新 m_i = 1 - (1 - m) * pi_i，与 MCRNCEAverage 完全一致；
    - 当 q 非空时，按逐位 m[i, l] = 1 - (1 - m) * pi_i * q[i, l] 更新并做整向量 L2 归一化；
    - 全程记录零范数事件；可选记录与同输入标量更新的方向余弦。
    """

    def __init__(self, base_nce: Any) -> None:
        super().__init__()
        self.base_nce = base_nce
        self.zero_norm_events_total: int = 0
        self.zero_norm_events_epoch: int = 0
        self.zero_v_norm_events_total: int = 0
        self.zero_b_norm_events_total: int = 0
        self.last_batch_zero_norm_events: int = 0
        self.scalar_direction_count_epoch: int = 0
        self.scalar_direction_cosine_sum_epoch: float = 0.0
        self.scalar_direction_cosine_min_epoch: Optional[float] = None

    @property
    def memory(self) -> torch.Tensor:
        return self.base_nce.memory

    @property
    def params(self) -> torch.Tensor:
        return self.base_nce.params

    @property
    def multinomial(self) -> Any:
        return self.base_nce.multinomial

    def reset_epoch_stats(self) -> None:
        self.zero_norm_events_epoch = 0
        self.scalar_direction_count_epoch = 0
        self.scalar_direction_cosine_sum_epoch = 0.0
        self.scalar_direction_cosine_min_epoch = None

    def scalar_direction_epoch_summary(self) -> Optional[Dict[str, Any]]:
        if self.scalar_direction_count_epoch == 0:
            return None
        return {
            "reference": "same-input scalar pi gate with L2 normalization",
            "count": int(self.scalar_direction_count_epoch),
            "cosine_mean": self.scalar_direction_cosine_sum_epoch / self.scalar_direction_count_epoch,
            "cosine_min": self.scalar_direction_cosine_min_epoch,
        }

    def forward(
        self,
        l: torch.Tensor,
        ab: torch.Tensor,
        y: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        pi: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        track_scalar_direction: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pi is None and q is None:
            self.last_batch_zero_norm_events = 0
            return self.base_nce(l, ab, y, idx=idx, epoch=epoch)

        base = self.base_nce
        K = int(base.params[0].item())
        T = float(base.params[1].item())
        Z_l = float(base.params[2].item())
        Z_ab = float(base.params[3].item())

        momentum = (
            float(base.params[4].item())
            if (epoch is None)
            else (0.0 if epoch < 0 else float(base.params[4].item()))
        )
        batch_size = l.size(0)
        output_size = base.memory.size(0)
        input_size = base.memory.size(1)

        if idx is None:
            drawn = base.multinomial.draw(batch_size * (base.K + 1))
            sample_idx: torch.Tensor = drawn.view(batch_size, -1)
            sample_idx.select(1, 0).copy_(y.data)
        else:
            sample_idx = idx

        if momentum <= 0:
            weight = (l + ab) / 2.0
            inx = torch.stack([torch.arange(batch_size)] * batch_size)
            inx = (
                torch.cat(
                    [
                        torch.arange(batch_size).view([-1, 1]),
                        inx[torch.eye(batch_size) == 0].view([batch_size, -1]),
                    ],
                    dim=1,
                )
                .to(weight.device)
                .view([-1])
            )
            weight = weight[inx].view([batch_size, batch_size, -1])
        else:
            weight = (
                torch.index_select(base.memory, 0, sample_idx.view(-1))
                .detach()
                .view(batch_size, K + 1, input_size)
            )

        weight = weight.sign_()
        out_ab = torch.bmm(weight, ab.view(batch_size, input_size, 1))
        out_l = torch.bmm(weight, l.view(batch_size, input_size, 1))

        if base.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            if Z_l < 0:
                base.params[2] = out_l.mean() * output_size
                Z_l = float(base.params[2].clone().detach().item())
            if Z_ab < 0:
                base.params[3] = out_ab.mean() * output_size
                Z_ab = float(base.params[3].clone().detach().item())
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        with torch.no_grad():
            l_pos_old = torch.index_select(base.memory, 0, y.view(-1))
            l_pos_new, _, stats = update_memory_bitwise(
                b_old=l_pos_old,
                l=l,
                ab=ab,
                momentum=momentum,
                pi=pi,
                q=q,
                compute_scalar_direction=track_scalar_direction,
            )
            base.memory.index_copy_(0, y, l_pos_new)
            z_cnt = int(stats["zero_norm_count"])
            self.last_batch_zero_norm_events = z_cnt
            self.zero_norm_events_epoch += z_cnt
            self.zero_norm_events_total += z_cnt
            self.zero_v_norm_events_total += int(stats["zero_v_norm_count"])
            self.zero_b_norm_events_total += int(stats["zero_b_norm_count"])
            direction_count = int(stats.get("scalar_direction_count", 0))
            if direction_count > 0:
                self.scalar_direction_count_epoch += direction_count
                self.scalar_direction_cosine_sum_epoch += float(stats["scalar_direction_cosine_sum"])
                direction_min = stats.get("scalar_direction_cosine_min")
                if direction_min is not None:
                    self.scalar_direction_cosine_min_epoch = (
                        float(direction_min)
                        if self.scalar_direction_cosine_min_epoch is None
                        else min(self.scalar_direction_cosine_min_epoch, float(direction_min))
                    )

        return out_l, out_ab


def _safe_roc_auc(bad_mask: np.ndarray, reliability_score: np.ndarray) -> Optional[float]:
    """计算用可信度分数区分正常样本 (~bad_mask) 与异常样本 (bad_mask) 的 ROC-AUC。"""
    n_bad = int(np.sum(bad_mask))
    n_total = int(bad_mask.shape[0])
    if not (0 < n_bad < n_total):
        return None
    try:
        return float(roc_auc_score((~bad_mask).astype(np.int32), reliability_score))
    except ValueError:
        return 0.5


def _safe_pr_auc_clean(bad_mask: np.ndarray, reliability_score: np.ndarray) -> Optional[float]:
    """以未置换/一致样本 (~bad_mask) 为正类，计算可信度分数的 PR-AUC (Average Precision)。"""
    n_bad = int(np.sum(bad_mask))
    n_total = int(bad_mask.shape[0])
    if not (0 < n_bad < n_total):
        return None
    try:
        return float(average_precision_score((~bad_mask).astype(np.int32), reliability_score))
    except ValueError:
        return None


def _safe_pr_auc_anomaly(bad_mask: np.ndarray, reliability_score: np.ndarray) -> Optional[float]:
    """以置换/标签不一致样本 (bad_mask) 为正类，计算 (1 - reliability_score) 的 PR-AUC。"""
    n_bad = int(np.sum(bad_mask))
    n_total = int(bad_mask.shape[0])
    if not (0 < n_bad < n_total):
        return None
    try:
        return float(average_precision_score(bad_mask.astype(np.int32), 1.0 - reliability_score))
    except ValueError:
        return None


def _quantiles_dict(arr: np.ndarray) -> Dict[str, float]:
    q10, q25, q50, q75, q90 = [
        float(v) for v in np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    ]
    return {"q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90": q90}


def _score_Candidate_summary(
    pi_arr: np.ndarray,
    noise_mask: np.ndarray,
    label_inconsistent_mask: np.ndarray,
) -> Dict[str, Any]:
    """汇总单个候选可信度分数的 ROC-AUC、PR-AUC、实际标签不一致 AUC、干净低可信尾部及分组均值。"""
    clean_sel = ~noise_mask
    permuted_sel = noise_mask
    label_consistent_sel = noise_mask & ~label_inconsistent_mask
    label_inconsistent_sel = label_inconsistent_mask

    return {
        "mean": float(np.mean(pi_arr)),
        "std": float(np.std(pi_arr)),
        "quantiles": _quantiles_dict(pi_arr),
        "roc_auc": _safe_roc_auc(noise_mask, pi_arr),
        "pr_auc": _safe_pr_auc_clean(noise_mask, pi_arr),
        "pr_auc_clean_pos": _safe_pr_auc_clean(noise_mask, pi_arr),
        "pr_auc_permuted_pos": _safe_pr_auc_anomaly(noise_mask, pi_arr),
        "label_inconsistent_auc": _safe_roc_auc(label_inconsistent_mask, pi_arr),
        "label_inconsistent_pr_auc": _safe_pr_auc_clean(label_inconsistent_mask, pi_arr),
        "label_inconsistent_pr_auc_anomaly_pos": _safe_pr_auc_anomaly(label_inconsistent_mask, pi_arr),
        "clean_low_pi_tail_lt_05": (
            float(np.mean(pi_arr[clean_sel] < 0.5)) if np.any(clean_sel) else None
        ),
        "clean_mean": float(np.mean(pi_arr[clean_sel])) if np.any(clean_sel) else None,
        "permuted_mean": float(np.mean(pi_arr[permuted_sel])) if np.any(permuted_sel) else None,
        "permuted_label_consistent_mean": (
            float(np.mean(pi_arr[label_consistent_sel])) if np.any(label_consistent_sel) else None
        ),
        "label_inconsistent_mean": (
            float(np.mean(pi_arr[label_inconsistent_sel])) if np.any(label_inconsistent_sel) else None
        ),
    }


def summarize_i2_diagnostics(
    i2_res: Dict[str, Any],
    noise_mask: np.ndarray,
    label_inconsistent_mask: np.ndarray,
    probe_c1_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """生成 M3 第 4.2 节要求的完整 I2 诊断字典。"""
    clean_sel = ~noise_mask
    permuted_sel = noise_mask
    label_consistent_sel = noise_mask & ~label_inconsistent_mask
    label_inconsistent_sel = label_inconsistent_mask

    c_bar = np.asarray(i2_res["c_bar"], dtype=np.float32)
    s_std = np.asarray(i2_res["s_std"], dtype=np.float32)
    pi_single = np.asarray(i2_res["pi_single"], dtype=np.float32)
    pi_mean = np.asarray(i2_res["pi_mean"], dtype=np.float32)
    pi_var = np.asarray(i2_res["pi_var"], dtype=np.float32)

    lambda_sens: Dict[str, Any] = {}
    for lam_key, arr in i2_res.get("pi_var_grid", {}).items():
        arr_np = np.asarray(arr, dtype=np.float32)
        cand = _score_Candidate_summary(arr_np, noise_mask, label_inconsistent_mask)
        cand["mean_abs_diff_vs_pi_mean"] = float(np.mean(np.abs(pi_mean - arr_np)))
        cand["max_abs_diff_vs_pi_mean"] = float(np.max(np.abs(pi_mean - arr_np)))
        lambda_sens[lam_key] = cand

    return {
        "k_views": int(i2_res["k_views"]),
        "lambda_var": float(i2_res["lambda_var"]),
        "gmm_fallback_single": bool(i2_res["gmm_fallback_single"]),
        "gmm_fallback_mean": bool(i2_res["gmm_fallback_mean"]),
        "single_gmm_means": [
            float(i2_res["single_gmm_clean_mean"]),
            float(i2_res["single_gmm_noisy_mean"]),
        ],
        "mean_gmm_means": [
            float(i2_res["mean_gmm_clean_mean"]),
            float(i2_res["mean_gmm_noisy_mean"]),
        ],
        "c_bar_distribution": {
            "mean": float(np.mean(c_bar)),
            "std": float(np.std(c_bar)),
            "quantiles": _quantiles_dict(c_bar),
            "clean_mean": float(np.mean(c_bar[clean_sel])) if np.any(clean_sel) else None,
            "permuted_mean": float(np.mean(c_bar[permuted_sel])) if np.any(permuted_sel) else None,
            "permuted_label_consistent_mean": (
                float(np.mean(c_bar[label_consistent_sel])) if np.any(label_consistent_sel) else None
            ),
            "label_inconsistent_mean": (
                float(np.mean(c_bar[label_inconsistent_sel])) if np.any(label_inconsistent_sel) else None
            ),
        },
        "s_distribution": {
            "mean": float(np.mean(s_std)),
            "std": float(np.std(s_std)),
            "quantiles": _quantiles_dict(s_std),
            "clean_mean": float(np.mean(s_std[clean_sel])) if np.any(clean_sel) else None,
            "clean_std": float(np.std(s_std[clean_sel])) if np.any(clean_sel) else None,
            "permuted_mean": float(np.mean(s_std[permuted_sel])) if np.any(permuted_sel) else None,
            "permuted_std": float(np.std(s_std[permuted_sel])) if np.any(permuted_sel) else None,
            "permuted_label_consistent_mean": (
                float(np.mean(s_std[label_consistent_sel])) if np.any(label_consistent_sel) else None
            ),
            "label_inconsistent_mean": (
                float(np.mean(s_std[label_inconsistent_sel])) if np.any(label_inconsistent_sel) else None
            ),
        },
        "pi_single": _score_Candidate_summary(pi_single, noise_mask, label_inconsistent_mask),
        "pi_mean": _score_Candidate_summary(pi_mean, noise_mask, label_inconsistent_mask),
        "pi_var": _score_Candidate_summary(pi_var, noise_mask, label_inconsistent_mask),
        "pi_mean_vs_var_diff": {
            "mean_abs_diff": float(np.mean(np.abs(pi_mean - pi_var))),
            "max_abs_diff": float(np.max(np.abs(pi_mean - pi_var))),
            "mean_signed_diff": float(np.mean(pi_mean - pi_var)),
        },
        "lambda_sensitivity": lambda_sens,
        "probe_c1_check": probe_c1_check,
    }


def summarize_i1_bitwise_diagnostics(
    pi_arr: np.ndarray,
    q_raw_arr: np.ndarray,
    q_active_arr: np.ndarray,
    momentum: float,
    noise_mask: np.ndarray,
    label_inconsistent_mask: np.ndarray,
    mem_signs_before: Optional[np.ndarray] = None,
    mem_signs_after: Optional[np.ndarray] = None,
    zero_norm_epoch: int = 0,
    zero_norm_total: int = 0,
    is_rand_q: bool = False,
    soft_beta: Optional[float] = None,
    soft_zero_mean_fallback_count: Optional[int] = None,
) -> Dict[str, Any]:
    """生成位级分布、写入预算、随机位与软门控守恒、位翻转和位均衡诊断。"""
    clean_sel = ~noise_mask
    permuted_sel = noise_mask
    label_consistent_sel = noise_mask & ~label_inconsistent_mask
    label_inconsistent_sel = label_inconsistent_mask

    pi_col = np.asarray(pi_arr, dtype=np.float64).reshape(-1, 1)
    q_raw = np.asarray(q_raw_arr, dtype=np.float64)
    q_act = np.asarray(q_active_arr, dtype=np.float64)

    one_minus_m = (1.0 - float(momentum)) * pi_col * q_act
    sample_write_budget = np.mean(one_minus_m, axis=1)
    sample_q_raw_mean = np.mean(q_raw, axis=1)
    sample_q_act_mean = np.mean(q_act, axis=1)

    sample_budget_raw = np.mean((1.0 - float(momentum)) * pi_col * q_raw, axis=1)
    q_mean_max_diff = float(np.max(np.abs(sample_q_act_mean - sample_q_raw_mean)))
    budget_max_diff = float(np.max(np.abs(sample_write_budget - sample_budget_raw)))

    non_const_rows = np.std(q_raw, axis=1) > 1e-8
    if np.any(non_const_rows):
        changed_rows = np.any(np.abs(q_act[non_const_rows] - q_raw[non_const_rows]) > 1e-8, axis=1)
        order_changed_frac: Optional[float] = float(np.mean(changed_rows))
    else:
        order_changed_frac = 0.0

    soft_gate_check: Optional[Dict[str, Any]] = None
    if soft_beta is not None:
        scalar_budget = ((1.0 - float(momentum)) * pi_col).reshape(-1)
        soft_gate_check = {
            "beta": float(soft_beta),
            "per_sample_gate_mean_max_abs_diff_from_one": float(np.max(np.abs(sample_q_act_mean - 1.0))),
            "per_sample_write_budget_max_abs_diff_vs_scalar": float(
                np.max(np.abs(sample_write_budget - scalar_budget))
            ),
            "min_gate": float(np.min(q_act)),
            "max_gate": float(np.max(q_act)),
            "zero_mean_fallback_count": (
                int(soft_zero_mean_fallback_count)
                if soft_zero_mean_fallback_count is not None
                else None
            ),
        }

    flip_stats: Optional[Dict[str, Any]] = None
    if mem_signs_before is not None and mem_signs_after is not None:
        flips = (mem_signs_before != mem_signs_after).astype(np.float32)
        sample_flip_rate = np.mean(flips, axis=1)
        flip_stats = {
            "overall_mean": float(np.mean(sample_flip_rate)),
            "clean_mean": float(np.mean(sample_flip_rate[clean_sel])) if np.any(clean_sel) else None,
            "permuted_mean": float(np.mean(sample_flip_rate[permuted_sel])) if np.any(permuted_sel) else None,
            "permuted_label_consistent_mean": (
                float(np.mean(sample_flip_rate[label_consistent_sel]))
                if np.any(label_consistent_sel)
                else None
            ),
            "label_inconsistent_mean": (
                float(np.mean(sample_flip_rate[label_inconsistent_sel]))
                if np.any(label_inconsistent_sel)
                else None
            ),
        }

    balance_stats: Optional[Dict[str, Any]] = None
    if mem_signs_after is not None:
        signs_f = np.asarray(mem_signs_after, dtype=np.float32)
        bit_means = np.mean(signs_f, axis=0)
        balance_stats = {
            "positive_bit_ratio": float(np.mean(signs_f > 0)),
            "zero_bit_ratio": float(np.mean(signs_f == 0)),
            "mean_abs_bit_bias": float(np.mean(np.abs(bit_means))),
            "max_abs_bit_bias": float(np.max(np.abs(bit_means))),
        }

    return {
        "q_distribution": {
            "mean": float(np.mean(q_raw)),
            "std": float(np.std(q_raw)),
            "quantiles": _quantiles_dict(q_raw.reshape(-1)),
            "sample_mean_quantiles": _quantiles_dict(sample_q_raw_mean),
            "clean_mean": float(np.mean(q_raw[clean_sel])) if np.any(clean_sel) else None,
            "permuted_mean": float(np.mean(q_raw[permuted_sel])) if np.any(permuted_sel) else None,
            "permuted_label_consistent_mean": (
                float(np.mean(q_raw[label_consistent_sel])) if np.any(label_consistent_sel) else None
            ),
            "label_inconsistent_mean": (
                float(np.mean(q_raw[label_inconsistent_sel])) if np.any(label_inconsistent_sel) else None
            ),
            "q_zero_rate_overall": float(np.mean(q_raw == 0.0)),
            "q_zero_rate_clean": float(np.mean(q_raw[clean_sel] == 0.0)) if np.any(clean_sel) else None,
            "q_zero_rate_permuted": (
                float(np.mean(q_raw[permuted_sel] == 0.0)) if np.any(permuted_sel) else None
            ),
            "q_zero_rate_label_inconsistent": (
                float(np.mean(q_raw[label_inconsistent_sel] == 0.0))
                if np.any(label_inconsistent_sel)
                else None
            ),
        },
        "write_budget_1_minus_m": {
            "momentum_reference": float(momentum),
            "overall_mean": float(np.mean(sample_write_budget)),
            "clean_mean": (
                float(np.mean(sample_write_budget[clean_sel])) if np.any(clean_sel) else None
            ),
            "permuted_mean": (
                float(np.mean(sample_write_budget[permuted_sel])) if np.any(permuted_sel) else None
            ),
            "permuted_label_consistent_mean": (
                float(np.mean(sample_write_budget[label_consistent_sel]))
                if np.any(label_consistent_sel)
                else None
            ),
            "label_inconsistent_mean": (
                float(np.mean(sample_write_budget[label_inconsistent_sel]))
                if np.any(label_inconsistent_sel)
                else None
            ),
        },
        "rand_q_check": {
            "is_rand_q": bool(is_rand_q),
            "per_sample_q_mean_max_abs_diff": q_mean_max_diff,
            "per_sample_write_budget_max_abs_diff": budget_max_diff,
            "non_constant_sample_order_changed_fraction": order_changed_frac,
        },
        "soft_gate_check": soft_gate_check,
        "memory_bit_flip_rate": flip_stats,
        "memory_bit_balance": balance_stats,
        "zero_norm_events_epoch": int(zero_norm_epoch),
        "zero_norm_events_total": int(zero_norm_total),
    }
