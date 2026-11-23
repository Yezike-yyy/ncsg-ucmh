"""NACG-Hash 实验运行目录与摘要完整性严格校验器。

在断点续跑跳过（SKIPPED）与子进程归档（COMPLETED）两个关口统一核验：
1. `summary.json` 与 `noise_mask.npy` 均为非空常规文件；
2. `noise_mask.npy` 可无 pickle 安全读取，为一维布尔数组，长度等于 `n_train`，
   `mask.sum() == n_noisy`，且与 `build_correspondence_permutation(n_train, noise_rate, 0)` 逐位一致；
3. 顶层与 `parsed_args` 的 `variant`、`seed`、`noise_seed == 0`、`noise_rate`、`bit`、
   `data_name`、`max_epochs == 20`、`lambda_graph`（及可选 `alpha`、`rch_v6_bridge_sharpen`）严格匹配；
4. `epoch_records` 长度为 20，逐轮字典含连续 `epoch == 0..19`、有限 `train_loss`、
   位于 `[0.0, 1.0]` 且算术一致的验证分数，`best_val_epoch` 与 `best_val_avg_official` 同逐轮最大值一致；
5. `eval_records` 为长度为 2 的双向评估字典列表，其 `map_official` 与顶层
   `test_i2t_official`、`test_t2i_official`、`test_avg_official` 均位于 `[0.0, 1.0]` 且彼此一致。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np  # type: ignore[import-not-found]

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from ours.mcr import build_correspondence_permutation  # type: ignore[import-not-found]


def _is_strict_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_num(val: Any) -> bool:
    return (not isinstance(val, bool)) and isinstance(val, (int, float)) and math.isfinite(float(val))


def _is_prob(val: Any) -> bool:
    return _is_finite_num(val) and (0.0 <= float(val) <= 1.0)


def validate_nacg_run_dir(
    run_dir: Path,
    *,
    expected_variant: str,
    expected_seed: int,
    expected_noise_rate: float,
    expected_bit: int,
    expected_data_name: str,
    expected_lambda_graph: float,
    expected_alpha: Optional[float] = None,
    expected_bridge_sharpen: Optional[float] = None,
    expected_use_fn_adjust: Optional[bool] = None,
    expected_max_epochs: int = 20,
    expected_noise_seed: int = 0,
) -> bool:
    """严格核验单个训练输出目录的摘要、逐轮轨迹、双向评估记录与噪声掩码数组。"""
    summary_path = run_dir / "summary.json"
    mask_path = run_dir / "noise_mask.npy"
    try:
        if not summary_path.is_file() or summary_path.stat().st_size <= 0:
            return False
        if not mask_path.is_file() or mask_path.stat().st_size <= 0:
            return False
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    if not isinstance(summary, dict):
        return False
    parsed_args = summary.get("parsed_args")
    if not isinstance(parsed_args, dict):
        return False
    # 正式 RCH 运行均未启用强制可靠度为一的退化对照。
    if parsed_args.get("nacg_force_one") is not False:
        return False

    # 1) 核心身份与超参数匹配校验
    if summary.get("variant") != expected_variant:
        return False
    if "variant" in parsed_args and parsed_args.get("variant") != expected_variant:
        return False

    raw_seed = summary.get("seed")
    if not _is_strict_int(raw_seed) or int(raw_seed) != int(expected_seed):
        return False
    if "seed" in parsed_args:
        arg_seed = parsed_args.get("seed")
        if not _is_strict_int(arg_seed) or int(arg_seed) != int(expected_seed):
            return False

    raw_noise_seed = summary.get("noise_seed")
    arg_noise_seed = parsed_args.get("noise_seed", raw_noise_seed)
    if not _is_strict_int(raw_noise_seed) or int(raw_noise_seed) != int(expected_noise_seed):
        return False
    if not _is_strict_int(arg_noise_seed) or int(arg_noise_seed) != int(expected_noise_seed):
        return False

    raw_nr = summary.get("noise_rate")
    arg_nr = parsed_args.get("noise_rate", raw_nr)
    if not _is_prob(raw_nr) or abs(float(raw_nr) - float(expected_noise_rate)) >= 1e-6:
        return False
    if not _is_prob(arg_nr) or abs(float(arg_nr) - float(expected_noise_rate)) >= 1e-6:
        return False

    raw_bit = summary.get("bit")
    arg_bit = parsed_args.get("bit")
    if not _is_strict_int(raw_bit) or int(raw_bit) != int(expected_bit):
        return False
    if not _is_strict_int(arg_bit) or int(arg_bit) != int(expected_bit):
        return False

    if summary.get("data_name") != expected_data_name:
        return False
    if parsed_args.get("data_name") != expected_data_name:
        return False

    arg_epochs = parsed_args.get("max_epochs")
    if not _is_strict_int(arg_epochs) or int(arg_epochs) != int(expected_max_epochs):
        return False

    raw_lam = parsed_args.get("lambda_graph")
    if not _is_finite_num(raw_lam) or abs(float(raw_lam) - float(expected_lambda_graph)) >= 1e-6:
        return False

    if expected_alpha is not None:
        raw_alpha = parsed_args.get("alpha")
        if not _is_finite_num(raw_alpha) or abs(float(raw_alpha) - float(expected_alpha)) >= 1e-6:
            return False

    if expected_bridge_sharpen is not None:
        raw_sharpen = parsed_args.get("rch_v6_bridge_sharpen")
        if not _is_finite_num(raw_sharpen) or abs(float(raw_sharpen) - float(expected_bridge_sharpen)) >= 1e-6:
            return False

    if expected_use_fn_adjust is not None:
        raw_fn_adj = parsed_args.get("rch_v6_use_fn_adjust")
        if not isinstance(raw_fn_adj, bool) or raw_fn_adj is not expected_use_fn_adjust:
            return False

    # 2) 噪声掩码数组形状、计数与固定种子置换逐位核验
    raw_n_train = summary.get("n_train")
    raw_n_noisy = summary.get("n_noisy")
    if not _is_strict_int(raw_n_train) or int(raw_n_train) <= 0:
        return False
    n_train = int(raw_n_train)
    if not _is_strict_int(raw_n_noisy) or not (0 <= int(raw_n_noisy) <= n_train):
        return False
    n_noisy = int(raw_n_noisy)

    try:
        mask_arr = np.load(mask_path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    if not isinstance(mask_arr, np.ndarray) or mask_arr.dtype != np.bool_ or mask_arr.ndim != 1:
        return False
    if int(mask_arr.shape[0]) != n_train or int(mask_arr.sum()) != n_noisy:
        return False
    expected_mask, _ = build_correspondence_permutation(
        n_train,
        float(expected_noise_rate),
        noise_seed=int(expected_noise_seed),
    )
    if not np.array_equal(mask_arr, expected_mask):
        return False

    # 3) 逐轮训练与验证轨迹完整性核验
    epoch_records = summary.get("epoch_records")
    if not isinstance(epoch_records, list) or len(epoch_records) != int(expected_max_epochs):
        return False

    val_avgs = []
    for ep_idx, rec in enumerate(epoch_records):
        if not isinstance(rec, dict):
            return False
        rec_ep = rec.get("epoch")
        if not _is_strict_int(rec_ep) or int(rec_ep) != ep_idx:
            return False
        if not _is_finite_num(rec.get("train_loss")):
            return False
        v_i2t = rec.get("val_i2t_official")
        v_t2i = rec.get("val_t2i_official")
        v_avg = rec.get("val_avg_official")
        if not (_is_prob(v_i2t) and _is_prob(v_t2i) and _is_prob(v_avg)):
            return False
        if abs(0.5 * (float(v_i2t) + float(v_t2i)) - float(v_avg)) > 1e-6:
            return False
        val_avgs.append(float(v_avg))

    best_ep = summary.get("best_val_epoch")
    best_val = summary.get("best_val_avg_official")
    if not _is_strict_int(best_ep) or not (0 <= int(best_ep) < int(expected_max_epochs)):
        return False
    if not _is_prob(best_val):
        return False
    if abs(float(best_val) - val_avgs[int(best_ep)]) > 1e-6:
        return False
    if abs(float(best_val) - max(val_avgs)) > 1e-6:
        return False

    # 4) 最终双向测试评估记录与顶层 MAP 指标一致性核验
    eval_records = summary.get("eval_records")
    if not isinstance(eval_records, list) or len(eval_records) != 2:
        return False
    if not isinstance(eval_records[0], dict) or not isinstance(eval_records[1], dict):
        return False

    ev_i2t = eval_records[0].get("map_official")
    ev_t2i = eval_records[1].get("map_official")
    top_i2t = summary.get("test_i2t_official")
    top_t2i = summary.get("test_t2i_official")
    top_avg = summary.get("test_avg_official")
    if not (_is_prob(ev_i2t) and _is_prob(ev_t2i) and _is_prob(top_i2t) and _is_prob(top_t2i) and _is_prob(top_avg)):
        return False
    if abs(float(ev_i2t) - float(top_i2t)) > 1e-6:
        return False
    if abs(float(ev_t2i) - float(top_t2i)) > 1e-6:
        return False
    if abs(0.5 * (float(top_i2t) + float(top_t2i)) - float(top_avg)) > 1e-6:
        return False

    return True
