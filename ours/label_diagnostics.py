"""文本置换后的多标签一致性事后诊断。"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def compute_label_inconsistency(
    labels: np.ndarray,
    noise_mask: np.ndarray,
    source_indices: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """按共享至少一个正标签的规则计算置换后的标签不一致掩码与统计。"""
    label_array = np.asarray(labels)
    mask_array = np.asarray(noise_mask)
    sources = np.asarray(source_indices)
    if label_array.ndim != 2:
        raise ValueError(f"labels必须是二维多标签数组，实际形状为{label_array.shape}")
    n_data = int(label_array.shape[0])
    if mask_array.shape != (n_data,) or mask_array.dtype != np.bool_:
        raise ValueError("noise_mask必须是与labels样本数一致的一维布尔数组")
    if sources.shape != (n_data,) or not np.issubdtype(sources.dtype, np.integer):
        raise ValueError("source_indices必须是与labels样本数一致的一维整数数组")
    if np.any(sources[~mask_array] != -1):
        raise ValueError("未置换位置的source_indices必须为-1")

    selected = np.flatnonzero(mask_array)
    selected_sources = sources[selected]
    if np.any((selected_sources < 0) | (selected_sources >= n_data)):
        raise ValueError("置换来源索引超出样本范围")
    if np.any(selected_sources == selected):
        raise ValueError("置换来源不能是样本自身")

    label_inconsistent_mask = np.zeros(n_data, dtype=bool)
    if selected.size:
        shared_positive = np.any(
            (label_array[selected] > 0) & (label_array[selected_sources] > 0),
            axis=1,
        )
        label_inconsistent_mask[selected] = ~shared_positive

    n_permuted = int(selected.size)
    n_inconsistent = int(label_inconsistent_mask.sum())
    return label_inconsistent_mask, {
        "diagnostic_scope": "post_hoc_only_not_used_for_training_or_gmm",
        "label_match_rule": "any((labels[i] > 0) & (labels[source_indices[i]] > 0))",
        "zero_label_handling": "置换对没有共享正标签即记为标签不一致",
        "label_shape": list(label_array.shape),
        "label_dtype": str(label_array.dtype),
        "n_train": n_data,
        "n_permuted": n_permuted,
        "permutation_fraction_observed": float(n_permuted / n_data) if n_data else 0.0,
        "n_label_inconsistent": n_inconsistent,
        "n_label_consistent_among_permuted": n_permuted - n_inconsistent,
        "actual_label_inconsistency_rate": float(n_inconsistent / n_data) if n_data else 0.0,
        "label_inconsistency_rate_among_permuted": float(n_inconsistent / n_permuted) if n_permuted else 0.0,
        "inconsistent_definition": "分母为全部训练样本；among_permuted分母仅为文本被置换的样本",
    }
