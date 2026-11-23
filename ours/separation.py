"""分离度门控与双分量可信度估计。"""
from __future__ import annotations

from typing import Any, Literal
import numpy as np  # type: ignore[import-not-found]
from sklearn.mixture import GaussianMixture  # type: ignore[import-not-found]


def fit_gmm_reliability_with_params(signals: np.ndarray, mode: Literal["consistency", "loss"] = "consistency", random_state: int = 1123) -> dict[str, Any]:
    """与原估计器使用相同拟合配置，并从同一模型提取参数。"""
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    lo, hi = float(np.min(sig)), float(np.max(sig))
    if hi - lo < 1e-12:
        return {"pi": np.ones(sig.shape[0], dtype=np.float32), "clean_mean": 0.0, "noisy_mean": 0.0, "fallback": True, "means": None, "variances": None, "weights": None, "D": None, "clean_component": None}
    norm = ((sig - lo) / (hi - lo)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=20, reg_covar=5e-4, tol=1e-2, random_state=int(random_state))
    gmm.fit(norm)
    means = np.asarray(gmm.means_, dtype=np.float64).reshape(-1)
    if mode == "consistency":
        clean = int(np.argmax(means))
    elif mode == "loss":
        clean = int(np.argmin(means))
    else:
        raise ValueError(f"未知信号模式: {mode}")
    variances = np.asarray(gmm.covariances_, dtype=np.float64)[:, 0, 0]
    return {"pi": gmm.predict_proba(norm)[:, clean].astype(np.float32), "clean_mean": float(means[clean]), "noisy_mean": float(means[1-clean]), "fallback": False, "means": means.tolist(), "variances": variances.tolist(), "weights": np.asarray(gmm.weights_).tolist(), "D": float(np.sqrt(2.0) * abs(means[0]-means[1]) / np.sqrt(variances.sum())), "clean_component": clean}


def apply_separation_gate(result: dict[str, Any], mode: str = "ashman", d_lo: float = 1.0, d_hi: float = 2.0, weight_thr: float = 0.85) -> tuple[np.ndarray, float | None]:
    """恒定信号回退为全一；端点值直接返回原数组以保留逐位退化。"""
    pi = np.asarray(result["pi"], dtype=np.float32)
    if result["fallback"]:
        return pi.copy(), None
    if mode == "ashman":
        if d_hi <= d_lo:
            raise ValueError("分离度上界必须大于下界")
        s = float(np.clip((result["D"] - d_lo) / (d_hi - d_lo), 0.0, 1.0))
    elif mode == "weight":
        s = 0.0 if result["weights"][result["clean_component"]] >= weight_thr else 1.0
    else:
        raise ValueError(f"未知门控模式: {mode}")
    if s == 1.0:
        return pi.copy(), s
    if s == 0.0:
        return np.ones_like(pi), s
    return (1.0 - s * (1.0 - pi)).astype(np.float32), s
