"""NACG-Hash（零分布校准与干净锚点桥接跨模态图哈希）核心算法模块。

包含三大解耦组件与两大训练家族接口：
  1. 模型无关冷启动锚点一致性（``build_nacg_anchor_state``）：
     利用预提取输入特征内积（VGG19 / BoW / Doc2Vec）在置换下对单模态内部结构 100% 免疫的物理不变量，
     借助全局锚点池（M_a=512，分块 O(N M_a) 并屏蔽自匹配对角列防止泄漏）与 3 步加权迭代，
     在 Epoch 0 训练前即生成高精度结构一致性得分 s_i^(0) 与 Top-K_b 干净跨模态桥接邻接表；
  2. 同批错配零分布校准（ODNC，``odnc_calibrate``）：
     仅以同批 M_0=4 组无不动点错配对照零分布 s_null 的均值/标准差标准化观测与零分布得分，
     通过左尾超额噪比（Storey-Efron）识别噪声比例 hat_eta：
       - 当 hat_eta == 0（干净数据集）时，严格门控为 r_i ≡ 1.0，实现 100% 逐位无损退化；
       - 当 hat_eta > 0 时，严格锁定噪声分量在零分布空间 (mu_0=0, sigma_0^2=1) 并运行约束 EM，
         从根本上杜绝对称 2-GMM 的双态翻转或组件坍缩；
  3. 干净锚点桥接目标重构与记忆库重合成：
     - Graph 插件家族（``jdsh_loss_nacg`` / ``djsrh_loss_nacg``）：保留 100% 单模态全量样本梯度，
       构建干净桥梁下的无污染单模态矩阵与非对称跨模态重构目标 S_{IT}^{rect}；
     - UCCH 主线家族（``NACGNCEAverage`` / ``nacg_ucch_contrastive_loss`` / ``nacg_positive_graph_loss``）：
       从 Epoch 0 起即以干净桥接语义写入标量恒定动量记忆库，消除垃圾锁定效应与假阴性排斥。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

from baselines.methods import djsrh_loss, jdsh_loss, jdsh_similarity  # type: ignore[import-not-found]
from ours.mcr import make_derangement  # type: ignore[import-not-found]


@dataclass
class NACGAnchorState:
    anchor_idx: torch.Tensor
    anchor_weights: torch.Tensor
    null_perms: torch.Tensor
    valid_mask: torch.Tensor
    norm_images: torch.Tensor
    norm_texts: torch.Tensor
    sim_anchor_i: torch.Tensor
    sim_anchor_t: torch.Tensor
    s_0: torch.Tensor
    s_0_null: torch.Tensor
    z_0: torch.Tensor
    z_0_null: torch.Tensor
    r_0: torch.Tensor
    r: torch.Tensor
    hat_eta_0: float
    hat_eta_raw_0: float
    cold_start_auc: Optional[float]
    nbr_idx_I: torch.Tensor
    nbr_w_I: torch.Tensor
    nbr_idx_T: torch.Tensor
    nbr_w_T: torch.Tensor
    num_anchors: int
    top_k_bridge: int
    tau_bridge: float
    tau_null: float
    quantiles: Tuple[float, ...]
    chunk_size: int
    verif_quantile: float = 0.35
    ridge_reg: float = 40.0
    mu_x: Optional[torch.Tensor] = None
    mu_y: Optional[torch.Tensor] = None
    idf_y: Optional[torch.Tensor] = None
    is_sparse_bow: bool = False
    px_norm: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    py_norm: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    self_consistent_storey: bool = False
    norm_texts_idf: Optional[torch.Tensor] = None


def _safe_roc_auc(noise_mask: Optional[np.ndarray], scores: np.ndarray) -> Optional[float]:
    if noise_mask is None:
        return None
    mask = np.asarray(noise_mask, dtype=bool).reshape(-1)
    n_noisy = int(mask.sum())
    if n_noisy <= 0 or n_noisy >= int(mask.shape[0]):
        return None
    try:
        return float(roc_auc_score((~mask).astype(np.int32), scores))
    except ValueError:
        return 0.5


def odnc_calibrate(
    s_obs: torch.Tensor,
    s_null: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_null: float = 0.12,
    quantiles: Tuple[float, ...] = (0.15, 0.25, 0.35),
    prev_r: Optional[torch.Tensor] = None,
    ema: float = 0.5,
    fixed_hat_eta: Optional[float] = None,
    self_consistent_storey: bool = False,
) -> Tuple[torch.Tensor, float, float, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """使用同批非对角无不动点零分布 s_null 校准观测得分 s_obs 并输出干净后验 r。

    重要不变量：必须严格仅用 s_null[:, valid_mask] 的均值 mu_0 与标准差 sigma_0 同时标准化
    s_obs 与 s_null，绝不使用 s_obs 自身均值/方差进行标准化。
    当 ``self_consistent_storey=True`` 时，保留 ``hat_eta_raw <= tau_null`` 的精确零门控，
    并在约束高斯 EM 内自洽扣除右分量在零分位点处的左尾重叠 F_1(t_q)，消除高分离度下的系统性低估。
    """
    n_samples = int(s_obs.shape[0])
    device = s_obs.device
    valid_count = int(valid_mask.sum().item())
    if valid_count <= 0:
        ones = torch.ones(n_samples, dtype=torch.float32, device=device)
        z_empty = torch.zeros_like(s_obs, dtype=torch.float32)
        z_null_empty = torch.zeros_like(s_null, dtype=torch.float32)
        return ones, 0.0, 0.0, z_empty, z_null_empty, {"zero_noise_exact_gate": True, "valid_count": 0}

    valid_null_1d = s_null[:, valid_mask].reshape(-1).to(torch.float32)
    mu_0 = float(valid_null_1d.mean().item())
    sigma_0 = max(float(valid_null_1d.std(unbiased=False).item()), 1e-6)
    z_obs = ((s_obs.to(torch.float32) - mu_0) / sigma_0).detach()
    z_null = ((s_null.to(torch.float32) - mu_0) / sigma_0).detach()

    valid_mask_np = valid_mask.detach().cpu().numpy().astype(bool)
    valid_obs_np = z_obs[valid_mask].detach().cpu().numpy().astype(np.float64)
    valid_null_np = z_null[:, valid_mask].detach().cpu().numpy().reshape(-1).astype(np.float64)

    ratios: List[float] = []
    q_stats: List[Tuple[float, float, float]] = []
    for q in quantiles:
        q_f = float(q)
        q_val = float(np.quantile(valid_null_np, q_f))
        frac = float(np.mean(valid_obs_np <= q_val))
        ratios.append(frac / q_f)
        q_stats.append((q_f, q_val, frac))
    hat_eta_raw = float(min(ratios)) if ratios else 0.0

    if fixed_hat_eta is None:
        if hat_eta_raw <= float(tau_null):
            hat_eta = 0.0
        else:
            hat_eta = float(np.clip((hat_eta_raw - float(tau_null)) / max(1.0 - float(tau_null), 1e-6), 0.0, 0.85))
    else:
        hat_eta = float(np.clip(float(fixed_hat_eta), 0.0, 0.85))

    if hat_eta <= 0.0:
        ones = torch.ones(n_samples, dtype=torch.float32, device=device)
        diag = {
            "zero_noise_exact_gate": True,
            "hat_eta_raw": hat_eta_raw,
            "hat_eta": 0.0,
            "null_mu_0": mu_0,
            "null_sigma_0": sigma_0,
            "clean_mu_z": None,
            "clean_var_z": None,
            "valid_count": valid_count,
            "self_consistent_storey": bool(self_consistent_storey),
        }
        return ones, 0.0, hat_eta_raw, z_obs, z_null, diag

    # 噪声高斯在 z 空间严格锁定 (mu_0=0, var_0=1)，仅迭代更新干净右分量 (mu_1 >= 0.5)
    w_0 = float(hat_eta)
    w_1 = 1.0 - w_0
    mu_0_z = 0.0
    var_0_z = 1.0

    thresh = float(np.quantile(valid_obs_np, w_0))
    top_vals = valid_obs_np[valid_obs_np >= thresh]
    if top_vals.size == 0:
        top_vals = valid_obs_np
    mu_1_z = max(float(np.mean(top_vals)), 0.8)
    var_1_z = float(np.clip(float(np.var(top_vals)), 0.25, 4.0))

    gamma_1 = np.full(valid_obs_np.shape[0], w_1, dtype=np.float64)
    log_2pi = math.log(2.0 * math.pi)
    em_iters = 10 if self_consistent_storey else 8
    for _ in range(em_iters):
        log_w0 = math.log(max(w_0, 1e-12))
        log_w1 = math.log(max(w_1, 1e-12))
        log_p0 = log_w0 - 0.5 * (log_2pi + math.log(var_0_z) + ((valid_obs_np - mu_0_z) ** 2) / var_0_z)
        log_p1 = log_w1 - 0.5 * (log_2pi + math.log(var_1_z) + ((valid_obs_np - mu_1_z) ** 2) / var_1_z)
        m_max = np.maximum(log_p0, log_p1)
        e0 = np.exp(log_p0 - m_max)
        e1 = np.exp(log_p1 - m_max)
        gamma_1 = e1 / np.maximum(e0 + e1, 1e-12)
        g_sum = float(np.sum(gamma_1))
        if g_sum > 1e-8:
            mu_1_z = max(float(np.sum(gamma_1 * valid_obs_np) / g_sum), 0.5)
            var_new = float(np.sum(gamma_1 * ((valid_obs_np - mu_1_z) ** 2)) / g_sum)
            var_1_z = float(np.clip(var_new, 0.15, 4.0))
        if self_consistent_storey and fixed_hat_eta is None and len(q_stats) > 0:
            inv_root2_s1 = 1.0 / math.sqrt(2.0 * var_1_z)
            eta_ests = []
            for q_f, t_q, frac_q in q_stats:
                f1_tq = 0.5 * (1.0 + math.erf((t_q - mu_1_z) * inv_root2_s1))
                eta_ests.append((frac_q - f1_tq) / max(q_f - f1_tq, 0.05))
            hat_eta = float(np.clip(float(np.median(eta_ests)), 0.0, 0.85))
            w_0 = hat_eta
            w_1 = 1.0 - w_0

    if self_consistent_storey and fixed_hat_eta is None:
        if hat_eta <= 0.0:
            ones = torch.ones(n_samples, dtype=torch.float32, device=device)
            diag = {
                "zero_noise_exact_gate": True,
                "hat_eta_raw": hat_eta_raw,
                "hat_eta": 0.0,
                "null_mu_0": mu_0,
                "null_sigma_0": sigma_0,
                "clean_mu_z": mu_1_z,
                "clean_var_z": var_1_z,
                "valid_count": valid_count,
                "self_consistent_storey": True,
            }
            return ones, 0.0, hat_eta_raw, z_obs, z_null, diag
        log_w0 = math.log(max(w_0, 1e-12))
        log_w1 = math.log(max(w_1, 1e-12))
        log_p0 = log_w0 - 0.5 * (log_2pi + math.log(var_0_z) + ((valid_obs_np - mu_0_z) ** 2) / var_0_z)
        log_p1 = log_w1 - 0.5 * (log_2pi + math.log(var_1_z) + ((valid_obs_np - mu_1_z) ** 2) / var_1_z)
        m_max = np.maximum(log_p0, log_p1)
        e0 = np.exp(log_p0 - m_max)
        e1 = np.exp(log_p1 - m_max)
        gamma_1 = e1 / np.maximum(e0 + e1, 1e-12)

    r_post_np = np.full(n_samples, w_1, dtype=np.float32)
    r_post_np[valid_mask_np] = gamma_1.astype(np.float32)
    r_post = torch.from_numpy(r_post_np).to(device=device, dtype=torch.float32)
    if prev_r is not None and not bool(torch.all(prev_r == 1.0).item()):
        r_out = (float(ema) * r_post + (1.0 - float(ema)) * prev_r.to(device=device, dtype=torch.float32)).clamp(0.0, 1.0)
    else:
        r_out = r_post.clamp(0.0, 1.0)

    diag = {
        "zero_noise_exact_gate": False,
        "hat_eta_raw": hat_eta_raw,
        "hat_eta": hat_eta,
        "null_mu_0": mu_0,
        "null_sigma_0": sigma_0,
        "clean_mu_z": mu_1_z,
        "clean_var_z": var_1_z,
        "valid_count": valid_count,
        "self_consistent_storey": bool(self_consistent_storey),
    }
    return r_out, hat_eta, hat_eta_raw, z_obs, z_null, diag


def _build_centered_anchor_kernels(
    images_gpu: torch.Tensor,
    texts_gpu: torch.Tensor,
    valid_mask: torch.Tensor,
    anchor_idx: torch.Tensor,
    chunk_size: int = 16384,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool]:
    """仅利用训练集有效行构造均值中心化（对稀疏非负文本自动启用 log1p*IDF）单位球面锚点核。

    [P2-2] 稀疏 BoW 约定说明：
      - 全局锚点核 ``sim_anchor_t`` 与 ``compute_centered_clean_graph``（旧版 RCH-v2）使用 ``mu_y`` 均值中心化，
        以便跨模态核 Ridge 回归 ``ux @ wx -> uy`` 在零均值特征空间求解（免除截距偏置）；
      - 在批内成对 Z-score 图 ``compute_zscore_clean_graph``（RCH-v3 / RCH-v4）中，当 ``is_sparse_bow=True`` 时，
        刻意跳过 ``- mu_y`` 稠密均值减法而直接对非负 ``log1p(y) * idf`` 归一化后再做非对角 Z-score 标准化，
        防止极短稀疏词袋向量（仅 1~2 个激活词）被稠密均值向量 ``-mu_y`` 稀释而产生伪共现相关。
    """
    device = images_gpu.device
    n_samples = int(images_gpu.shape[0])
    m_anc = int(anchor_idx.shape[0])

    xv = images_gpu[valid_mask].to(torch.float32)
    yv = texts_gpu[valid_mask].to(torch.float32)
    mu_x = xv.mean(dim=0, keepdim=True)
    zero_entry_frac = float((yv.abs() <= 1e-6).float().mean().item()) if yv.numel() > 0 else 0.0
    is_sparse_bow = (yv.numel() > 0) and (float(yv.min().item()) >= -1e-6) and (zero_entry_frac > 0.5)
    if is_sparse_bow:
        df = (yv.abs() > 1e-6).float().mean(dim=0, keepdim=True).clamp_min(1.0 / max(n_samples, 1))
        idf: Optional[torch.Tensor] = torch.log(1.0 + 1.0 / df)
        mu_y = (torch.log1p(yv.clamp_min(0.0)) * idf).mean(dim=0, keepdim=True)
    else:
        idf = None
        mu_y = yv.mean(dim=0, keepdim=True)

    anc_x_raw = images_gpu.index_select(0, anchor_idx).to(torch.float32)
    anc_xc = F.normalize(anc_x_raw - mu_x, dim=1, eps=1e-8)
    anc_y_raw = texts_gpu.index_select(0, anchor_idx).to(torch.float32)
    if idf is not None:
        anc_y_raw = torch.log1p(anc_y_raw.clamp_min(0.0)) * idf
    anc_yc = F.normalize(anc_y_raw - mu_y, dim=1, eps=1e-8)

    sim_anchor_i = torch.empty((n_samples, m_anc), device=device, dtype=torch.float32)
    sim_anchor_t = torch.empty((n_samples, m_anc), device=device, dtype=torch.float32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        xc_chunk = F.normalize(images_gpu[start:end].to(torch.float32) - mu_x, dim=1, eps=1e-8)
        y_chunk = texts_gpu[start:end].to(torch.float32)
        if idf is not None:
            y_chunk = torch.log1p(y_chunk.clamp_min(0.0)) * idf
        yc_chunk = F.normalize(y_chunk - mu_y, dim=1, eps=1e-8)
        sim_anchor_i[start:end] = xc_chunk @ anc_xc.t()
        sim_anchor_t[start:end] = yc_chunk @ anc_yc.t()

    idf_out = idf.detach() if idf is not None else None
    return sim_anchor_i, sim_anchor_t, mu_x.detach(), mu_y.detach(), idf_out, bool(is_sparse_bow)


def _crossfit_cold_start_calibrate(
    sim_anchor_i: torch.Tensor,
    sim_anchor_t: torch.Tensor,
    valid_mask: torch.Tensor,
    anchor_idx: torch.Tensor,
    null_perms: torch.Tensor,
    rng: np.random.RandomState,
    refine_steps: int = 3,
    tau_null: float = 0.12,
    quantiles: Tuple[float, ...] = (0.15, 0.25, 0.35),
    verif_quantile: float = 0.35,
    ridge_reg: float = 40.0,
    chunk_size: int = 16384,
    img_valid_mask: Optional[torch.Tensor] = None,
    self_consistent_storey: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    torch.Tensor,
    torch.Tensor,
]:
    """2-fold 折外锚点核加权 Ridge 交叉预测与边际可预测子集 ODNC 校准（局部 rng，屏蔽自匹配）。"""
    device = sim_anchor_i.device
    n_samples, m_anc = int(sim_anchor_i.shape[0]), int(sim_anchor_i.shape[1])

    ux_raw = sim_anchor_i.clone()
    uy_raw = sim_anchor_t.clone()
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        row_idx = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(1)
        self_mask = row_idx == anchor_idx.unsqueeze(0)
        ux_raw[start:end].masked_fill_(self_mask, 0.0)
        uy_raw[start:end].masked_fill_(self_mask, 0.0)

    ux = ux_raw - ux_raw[valid_mask].mean(dim=0, keepdim=True)
    uy = uy_raw - uy_raw[valid_mask].mean(dim=0, keepdim=True)
    uxn = F.normalize(ux, dim=1, eps=1e-8)
    uyn = F.normalize(uy, dim=1, eps=1e-8)

    v_dev = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    n_valid = int(v_dev.shape[0])
    if n_valid < 4:
        folds = [(v_dev, v_dev)]
    else:
        fold_perm = torch.from_numpy(rng.permutation(n_valid).astype(np.int64)).to(device=device)
        v_shuffled = v_dev.index_select(0, fold_perm)
        half = n_valid // 2
        f0 = v_shuffled[:half]
        f1 = v_shuffled[half:]
        folds = [(f0, f1), (f1, f0)]

    eye = torch.eye(m_anc, device=device, dtype=torch.float32) * float(ridge_reg)
    r_curr = torch.ones(n_samples, device=device, dtype=torch.float32)
    s_obs = torch.zeros(n_samples, device=device, dtype=torch.float32)
    s_null = torch.zeros((int(null_perms.shape[0]), n_samples), device=device, dtype=torch.float32)
    z_0 = torch.zeros_like(s_obs)
    z_0_null = torch.zeros_like(s_null)
    hat_eta_0 = 0.0
    hat_eta_raw_0 = 0.0
    px_last = torch.zeros((n_samples, m_anc), device=device, dtype=torch.float32)
    py_last = torch.zeros((n_samples, m_anc), device=device, dtype=torch.float32)
    wx_last: List[torch.Tensor] = []

    steps = max(int(refine_steps), 1)
    for step in range(steps):
        nx = torch.zeros(n_samples, device=device, dtype=torch.float32)
        ny = torch.zeros(n_samples, device=device, dtype=torch.float32)
        px_all = torch.zeros((n_samples, m_anc), device=device, dtype=torch.float32)
        py_all = torch.zeros((n_samples, m_anc), device=device, dtype=torch.float32)
        wx_step: List[torch.Tensor] = []

        for tr, te in folds:
            w = r_curr.index_select(0, tr).unsqueeze(1)
            ux_tr = ux.index_select(0, tr)
            uy_tr = uy.index_select(0, tr)
            wx = torch.linalg.solve(ux_tr.t() @ (ux_tr * w) + eye, ux_tr.t() @ (uy_tr * w))
            wy = torch.linalg.solve(uy_tr.t() @ (uy_tr * w) + eye, uy_tr.t() @ (ux_tr * w))
            wx_step.append(wx)

            px_raw_te = ux.index_select(0, te) @ wx
            py_raw_te = uy.index_select(0, te) @ wy
            nx.index_copy_(0, te, px_raw_te.norm(dim=1))
            ny.index_copy_(0, te, py_raw_te.norm(dim=1))
            px_all.index_copy_(0, te, F.normalize(px_raw_te, dim=1, eps=1e-8))
            py_all.index_copy_(0, te, F.normalize(py_raw_te, dim=1, eps=1e-8))

        px_last = px_all
        py_last = py_all
        wx_last = wx_step

        s_obs = 0.5 * ((px_all * uyn).sum(dim=1) + (py_all * uxn).sum(dim=1))
        s_null = torch.stack(
            [
                0.5 * ((px_all * uyn.index_select(0, pm)).sum(dim=1) + (py_all.index_select(0, pm) * uxn).sum(dim=1))
                for pm in null_perms
            ],
            dim=0,
        )

        if verif_quantile > 0.0 and n_valid >= 16:
            q_nx = torch.quantile(nx[valid_mask], float(verif_quantile))
            q_ny = torch.quantile(ny[valid_mask], float(verif_quantile))
            verif_mask = valid_mask & (nx >= q_nx) & (ny >= q_ny)
            if int(verif_mask.sum().item()) < 8:
                verif_mask = valid_mask
        else:
            verif_mask = valid_mask

        _, eta_step, eta_raw_step, _, _, _ = odnc_calibrate(
            s_obs=s_obs,
            s_null=s_null,
            valid_mask=verif_mask,
            tau_null=tau_null,
            quantiles=quantiles,
            self_consistent_storey=self_consistent_storey,
        )
        r_step, _, _, z_step, z_null_step, _ = odnc_calibrate(
            s_obs=s_obs,
            s_null=s_null,
            valid_mask=valid_mask,
            tau_null=tau_null,
            quantiles=quantiles,
            fixed_hat_eta=eta_step,
            self_consistent_storey=self_consistent_storey,
        )
        z_0, z_0_null, r_curr = z_step, z_null_step, r_step
        hat_eta_0, hat_eta_raw_0 = eta_step, eta_raw_step
        if step == 0 and hat_eta_0 <= 0.0:
            r_curr = torch.ones(n_samples, device=device, dtype=torch.float32)
            break

    if hat_eta_0 <= 0.0:
        r_curr = torch.ones(n_samples, device=device, dtype=torch.float32)

    px_norm = px_last.clone()
    if img_valid_mask is not None and len(wx_last) > 0:
        extra_img_mask = img_valid_mask & (~valid_mask)
        if bool(extra_img_mask.any().item()):
            extra_idx = torch.nonzero(extra_img_mask, as_tuple=False).view(-1)
            wx_mean = torch.stack(wx_last, dim=0).mean(dim=0)
            px_extra = F.normalize(ux.index_select(0, extra_idx) @ wx_mean, dim=1, eps=1e-8)
            px_norm.index_copy_(0, extra_idx, px_extra)
        px_norm = px_norm * img_valid_mask.unsqueeze(1).to(dtype=px_norm.dtype)
    py_norm = (py_last * valid_mask.unsqueeze(1).to(dtype=py_last.dtype)).contiguous()
    px_norm = px_norm.contiguous()

    return s_obs, s_null, z_0, z_0_null, r_curr, hat_eta_0, hat_eta_raw_0, px_norm.detach(), py_norm.detach()


def build_bridge_neighbors(
    sim_anchor_i: torch.Tensor,
    sim_anchor_t: torch.Tensor,
    anchor_idx: torch.Tensor,
    r_full: torch.Tensor,
    top_k_bridge: int = 16,
    tau_bridge: float = 0.07,
    chunk_size: int = 16384,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """分块构造每个样本在中心化对齐单模态锚点核下的 Top-K_b 干净锚点检索索引与桥接权重。"""
    n_samples = int(sim_anchor_i.shape[0])
    m_anc = int(anchor_idx.shape[0])
    k_b = min(int(top_k_bridge), max(m_anc - 1, 1))
    device = sim_anchor_i.device
    log_r_anc = 2.0 * torch.log(r_full.index_select(0, anchor_idx).clamp_min(1e-4)).unsqueeze(0)

    nbr_idx_i = torch.empty((n_samples, k_b), device=device, dtype=torch.long)
    nbr_w_i = torch.empty((n_samples, k_b), device=device, dtype=torch.float32)
    nbr_idx_t = torch.empty((n_samples, k_b), device=device, dtype=torch.long)
    nbr_w_t = torch.empty((n_samples, k_b), device=device, dtype=torch.float32)

    inv_tau = 1.0 / max(float(tau_bridge), 1e-4)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        row_idx = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(1)
        self_mask = row_idx == anchor_idx.unsqueeze(0)

        score_i = (sim_anchor_i[start:end] * inv_tau + log_r_anc).masked_fill(self_mask, -1e9)
        top_vi, top_ii = torch.topk(score_i, k=k_b, dim=1)
        nbr_idx_i[start:end] = top_ii
        nbr_w_i[start:end] = F.softmax(top_vi, dim=1)

        score_t = (sim_anchor_t[start:end] * inv_tau + log_r_anc).masked_fill(self_mask, -1e9)
        top_vt, top_it = torch.topk(score_t, k=k_b, dim=1)
        nbr_idx_t[start:end] = top_it
        nbr_w_t[start:end] = F.softmax(top_vt, dim=1)

    return nbr_idx_i, nbr_w_i, nbr_idx_t, nbr_w_t


def build_nacg_anchor_state(
    images_gpu: torch.Tensor,
    texts_gpu: torch.Tensor,
    seed: int,
    num_anchors: int = 512,
    num_null: int = 4,
    refine_steps: int = 3,
    top_k_bridge: int = 16,
    tau_bridge: float = 0.07,
    tau_null: float = 0.12,
    quantiles: Tuple[float, ...] = (0.15, 0.25, 0.35),
    noise_mask: Optional[np.ndarray] = None,
    chunk_size: int = 16384,
    verif_quantile: float = 0.35,
    ridge_reg: float = 40.0,
    store_full_norm: bool = True,
    self_consistent_storey: bool = False,
) -> NACGAnchorState:
    """仅用局部独立 rng 在 Epoch 0 开始前构造 NACG 冷启动状态（不消耗全局随机数流）。"""
    with torch.no_grad():
        device = images_gpu.device
        n_samples = int(images_gpu.shape[0])
        rng = np.random.RandomState(int(seed) + 2026)

        img_norm = images_gpu.to(torch.float32).norm(dim=1)
        txt_norm = texts_gpu.to(torch.float32).norm(dim=1)
        img_valid_mask = img_norm > 1e-6
        valid_mask = img_valid_mask & (txt_norm > 1e-6)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1).detach().cpu().numpy()
        if int(valid_indices.shape[0]) < 4:
            valid_indices = np.arange(n_samples, dtype=np.int64)
            valid_mask = torch.ones(n_samples, dtype=torch.bool, device=device)
            img_valid_mask = valid_mask

        if store_full_norm:
            norm_images = F.normalize(images_gpu.to(torch.float32), dim=1, eps=1e-8).contiguous()
            norm_texts = F.normalize(texts_gpu.to(torch.float32), dim=1, eps=1e-8).contiguous()
        else:
            norm_images = torch.empty((0,), device=device, dtype=torch.float32)
            norm_texts = torch.empty((0,), device=device, dtype=torch.float32)

        m_anc = min(int(num_anchors), int(valid_indices.shape[0]))
        chosen = rng.choice(valid_indices, size=m_anc, replace=False).astype(np.int64)
        anchor_idx = torch.from_numpy(chosen).to(device=device, dtype=torch.long)

        perm_list = [make_derangement(rng, n_samples) for _ in range(int(num_null))]
        null_perms = torch.from_numpy(np.stack(perm_list, axis=0)).to(device=device, dtype=torch.long)

        sim_anchor_i, sim_anchor_t, mu_x, mu_y, idf_y, is_sparse_bow = _build_centered_anchor_kernels(
            images_gpu=images_gpu,
            texts_gpu=texts_gpu,
            valid_mask=valid_mask,
            anchor_idx=anchor_idx,
            chunk_size=chunk_size,
        )

        norm_texts_idf: Optional[torch.Tensor] = None
        if store_full_norm and is_sparse_bow and idf_y is not None:
            norm_texts_idf_buf = torch.empty((n_samples, int(texts_gpu.shape[1])), device=device, dtype=torch.float32)
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                y_c = torch.log1p(texts_gpu[start:end].to(torch.float32).clamp_min(0.0)) * idf_y
                v_c = valid_mask[start:end].unsqueeze(1).to(torch.float32)
                norm_texts_idf_buf[start:end] = F.normalize(y_c, dim=1, eps=1e-8) * v_c
            norm_texts_idf = norm_texts_idf_buf.contiguous()

        s_obs, s_null, z_0, z_0_null, r_0, hat_eta_0, hat_eta_raw_0, px_norm, py_norm = _crossfit_cold_start_calibrate(
            sim_anchor_i=sim_anchor_i,
            sim_anchor_t=sim_anchor_t,
            valid_mask=valid_mask,
            anchor_idx=anchor_idx,
            null_perms=null_perms,
            rng=rng,
            refine_steps=refine_steps,
            tau_null=tau_null,
            quantiles=quantiles,
            verif_quantile=verif_quantile,
            ridge_reg=ridge_reg,
            chunk_size=chunk_size,
            img_valid_mask=img_valid_mask,
            self_consistent_storey=self_consistent_storey,
        )
        anchor_weights = r_0.index_select(0, anchor_idx).clamp_min(0.05)

        cold_auc = _safe_roc_auc(noise_mask, r_0.detach().cpu().numpy() if hat_eta_0 > 0.0 else s_obs.detach().cpu().numpy())
        nbr_idx_i, nbr_w_i, nbr_idx_t, nbr_w_t = build_bridge_neighbors(
            sim_anchor_i=sim_anchor_i,
            sim_anchor_t=sim_anchor_t,
            anchor_idx=anchor_idx,
            r_full=r_0,
            top_k_bridge=top_k_bridge,
            tau_bridge=tau_bridge,
            chunk_size=chunk_size,
        )
        return NACGAnchorState(
            anchor_idx=anchor_idx,
            anchor_weights=anchor_weights,
            null_perms=null_perms,
            valid_mask=valid_mask,
            norm_images=norm_images,
            norm_texts=norm_texts,
            sim_anchor_i=sim_anchor_i,
            sim_anchor_t=sim_anchor_t,
            s_0=s_obs,
            s_0_null=s_null,
            z_0=z_0,
            z_0_null=z_0_null,
            r_0=r_0,
            r=r_0.clone(),
            hat_eta_0=float(hat_eta_0),
            hat_eta_raw_0=float(hat_eta_raw_0),
            cold_start_auc=cold_auc,
            nbr_idx_I=nbr_idx_i,
            nbr_w_I=nbr_w_i,
            nbr_idx_T=nbr_idx_t,
            nbr_w_T=nbr_w_t,
            num_anchors=m_anc,
            top_k_bridge=int(nbr_idx_i.shape[1]),
            tau_bridge=float(tau_bridge),
            tau_null=float(tau_null),
            quantiles=tuple(float(q) for q in quantiles),
            chunk_size=int(chunk_size),
            verif_quantile=float(verif_quantile),
            ridge_reg=float(ridge_reg),
            mu_x=mu_x,
            mu_y=mu_y,
            idf_y=idf_y,
            is_sparse_bow=is_sparse_bow,
            px_norm=px_norm,
            py_norm=py_norm,
            self_consistent_storey=bool(self_consistent_storey),
            norm_texts_idf=norm_texts_idf,
        )


def update_nacg_epoch_reliability(
    state: NACGAnchorState,
    hx_all: torch.Tensor,
    hy_all: torch.Tensor,
    epoch: int,
    noise_mask: Optional[np.ndarray] = None,
    force_one: bool = False,
    online_dominant: bool = False,
    self_consistent_storey: bool = False,
    v5_mode: bool = False,
    refresh_bridges: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """根据当轮完整归一化哈希表示耦合冷启动先验并经 ODNC 更新可信度与干净锚点桥接索引。"""
    with torch.no_grad():
        n_samples = int(state.valid_mask.shape[0])
        if force_one or state.hat_eta_0 <= 0.0:
            state.r = torch.ones(n_samples, device=state.valid_mask.device, dtype=torch.float32)
            r_np = state.r.detach().cpu().numpy()
            auc = _safe_roc_auc(noise_mask, r_np)
            diag: Dict[str, Any] = {
                "pi_mean": 1.0,
                "pi_quantiles": [1.0, 1.0, 1.0, 1.0, 1.0],
                "pi_auc": auc,
                "pi_clean_mean": 1.0,
                "pi_noisy_mean": 1.0 if (noise_mask is not None and bool(np.any(noise_mask))) else None,
                "hat_eta_0": float(state.hat_eta_0),
                "hat_eta_raw_0": float(state.hat_eta_raw_0),
                "hat_eta": 0.0,
                "cold_start_auc": state.cold_start_auc,
                "zero_noise_exact_gate": True,
                "lam": 0.0,
                "self_consistent_storey": bool(self_consistent_storey),
                "v5_mode": bool(v5_mode),
                "refresh_bridges": bool(refresh_bridges),
            }
            return state.r, diag

        hx_n = F.normalize(hx_all.to(device=state.z_0.device, dtype=torch.float32), dim=1, eps=1e-8)
        hy_n = F.normalize(hy_all.to(device=state.z_0.device, dtype=torch.float32), dim=1, eps=1e-8)
        c_obs = (hx_n * hy_n).sum(dim=1)
        c_null = torch.stack([(hx_n * hy_n.index_select(0, perm)).sum(dim=1) for perm in state.null_perms], dim=0)

        c_valid_null = c_null[:, state.valid_mask].reshape(-1)
        mu_c0 = float(c_valid_null.mean().item())
        sigma_c0 = max(float(c_valid_null.std(unbiased=False).item()), 1e-6)
        z_c_obs = (c_obs - mu_c0) / sigma_c0
        z_c_null = (c_null - mu_c0) / sigma_c0

        cal_sc = bool(self_consistent_storey) and (not v5_mode)
        if v5_mode:
            progress = min(max(float(epoch), 0.0) / 10.0, 1.0)
            lam = min(0.65, 0.50 + 0.15 * progress)
            s_fused = (1.0 - lam) * state.z_0 + lam * z_c_obs
            s_fused_null = (1.0 - lam) * state.z_0_null + lam * z_c_null
            _, eta_online, _, _, _, _ = odnc_calibrate(
                s_obs=s_fused,
                s_null=s_fused_null,
                valid_mask=state.valid_mask,
                tau_null=state.tau_null,
                quantiles=state.quantiles,
                self_consistent_storey=False,
            )
            eta_floor = min(0.45, 0.88 * float(state.hat_eta_raw_0))
            eta_target = max(float(state.hat_eta_0), float(eta_floor), float(eta_online))
            prev_r_arg = None if epoch == 0 else state.r
            ema_val = 0.6
        elif self_consistent_storey:
            progress = min(max(float(epoch), 0.0) / 10.0, 1.0)
            lam = min(0.65, 0.45 + 0.20 * progress)
            s_fused = (1.0 - lam) * state.z_0 + lam * z_c_obs
            s_fused_null = (1.0 - lam) * state.z_0_null + lam * z_c_null
            _, eta_online, _, _, _, _ = odnc_calibrate(
                s_obs=s_fused,
                s_null=s_fused_null,
                valid_mask=state.valid_mask,
                tau_null=state.tau_null,
                quantiles=state.quantiles,
                self_consistent_storey=True,
            )
            eta_target = max(float(state.hat_eta_0), float(eta_online))
            prev_r_arg = None if (online_dominant and epoch == 0) else state.r
            ema_val = 0.6
        elif online_dominant:
            lam = min(0.88, 0.72 + 0.04 * float(epoch))
            s_fused = (1.0 - lam) * state.z_0 + lam * z_c_obs
            s_fused_null = (1.0 - lam) * state.z_0_null + lam * z_c_null
            _, eta_online, _, _, _, _ = odnc_calibrate(
                s_obs=s_fused,
                s_null=s_fused_null,
                valid_mask=state.valid_mask,
                tau_null=state.tau_null,
                quantiles=state.quantiles,
                self_consistent_storey=False,
            )
            eta_target = max(float(state.hat_eta_0), float(eta_online))
            prev_r_arg = None if epoch == 0 else state.r
            ema_val = 0.6
        else:
            lam = min(0.65, float(epoch + 1) / 10.0)
            s_fused = (1.0 - lam) * state.z_0 + lam * z_c_obs
            s_fused_null = (1.0 - lam) * state.z_0_null + lam * z_c_null
            eta_target = float(state.hat_eta_0)
            prev_r_arg = state.r
            ema_val = 0.5

        r_new, hat_eta, hat_eta_raw, _, _, cal_diag = odnc_calibrate(
            s_obs=s_fused,
            s_null=s_fused_null,
            valid_mask=state.valid_mask,
            tau_null=state.tau_null,
            quantiles=state.quantiles,
            prev_r=prev_r_arg,
            ema=ema_val,
            fixed_hat_eta=eta_target,
            self_consistent_storey=cal_sc,
        )
        state.r = r_new
        if refresh_bridges:
            nbr_idx_i, nbr_w_i, nbr_idx_t, nbr_w_t = build_bridge_neighbors(
                sim_anchor_i=state.sim_anchor_i,
                sim_anchor_t=state.sim_anchor_t,
                anchor_idx=state.anchor_idx,
                r_full=state.r,
                top_k_bridge=state.top_k_bridge,
                tau_bridge=state.tau_bridge,
                chunk_size=state.chunk_size,
            )
            state.nbr_idx_I = nbr_idx_i
            state.nbr_w_I = nbr_w_i
            state.nbr_idx_T = nbr_idx_t
            state.nbr_w_T = nbr_w_t

        r_np = state.r.detach().cpu().numpy()
        auc = _safe_roc_auc(noise_mask, r_np)
        pi_clean = None
        pi_noisy = None
        if noise_mask is not None:
            nm = np.asarray(noise_mask, dtype=bool)
            clean = ~nm
            pi_clean = float(np.mean(r_np[clean])) if bool(np.any(clean)) else None
            pi_noisy = float(np.mean(r_np[nm])) if bool(np.any(nm)) else None

        diag = {
            "pi_mean": float(np.mean(r_np)),
            "pi_quantiles": np.quantile(r_np, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
            "pi_auc": auc,
            "pi_clean_mean": pi_clean,
            "pi_noisy_mean": pi_noisy,
            "hat_eta_0": float(state.hat_eta_0),
            "hat_eta_raw_0": float(state.hat_eta_raw_0),
            "hat_eta": float(hat_eta),
            "hat_eta_raw": float(hat_eta_raw),
            "cold_start_auc": state.cold_start_auc,
            "zero_noise_exact_gate": False,
            "lam": float(lam),
            "clean_mu_z": cal_diag.get("clean_mu_z"),
            "clean_var_z": cal_diag.get("clean_var_z"),
            "self_consistent_storey": cal_sc,
            "v5_mode": bool(v5_mode),
            "refresh_bridges": bool(refresh_bridges),
        }
        return state.r, diag


def gather_bridged_features(
    anchor_feats: torch.Tensor,
    nbr_idx_batch: torch.Tensor,
    nbr_w_batch: torch.Tensor,
) -> torch.Tensor:
    """根据每个样本的 (B, K_b) 锚点邻居索引与权重合成单位球面桥接向量。"""
    gathered = anchor_feats.index_select(0, nbr_idx_batch.reshape(-1)).view(
        int(nbr_idx_batch.shape[0]),
        int(nbr_idx_batch.shape[1]),
        int(anchor_feats.shape[1]),
    )
    pooled = (gathered * nbr_w_batch.unsqueeze(-1).to(gathered.dtype)).sum(dim=1)
    return F.normalize(pooled, dim=1, eps=1e-8)


def _jdsh_stretch(similarity: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    left = params["loc_left"] - params["alpha_left"] * params["scale_left"]
    right = params["loc_right"] + params["beta_right"] * params["scale_right"]
    low = similarity < left
    high_mask = similarity > right
    stretched = similarity.clone()
    stretched[low] = (1.0 + params["l1"] * torch.exp(-(stretched[low] - params["min"]))) * stretched[low]
    stretched[high_mask] = (1.0 + params["l2"] * torch.exp(stretched[high_mask] - params["max"])) * stretched[high_mask]
    return stretched * params["mu"]


def jdsh_similarity_nacg(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    params: Dict[str, Any],
    r_batch: torch.Tensor,
    x_bridge: torch.Tensor,
    y_bridge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建 JDSH 无污染单模态相似目标 (sim_ii, sim_tt) 与干净桥接非对称跨模态目标 sim_it。

    注：``x_bridge`` 源自干净文本近邻对图像的锚点聚合（填补错配文本的真实图像语义）；
    ``y_bridge`` 源自干净图像近邻对文本的锚点聚合（填补错配图像的真实文本语义）。
    """
    im = F.normalize(image_features)
    tx = F.normalize(texts)
    r_flat = r_batch.detach().to(dtype=im.dtype, device=im.device).view(-1)
    r_col = r_flat.unsqueeze(1)
    r_row = r_flat.unsqueeze(0)

    y_for_i = F.normalize(r_col * tx + (1.0 - r_col) * y_bridge.detach().to(im.dtype), dim=1)
    x_for_t = F.normalize(r_col * im + (1.0 - r_col) * x_bridge.detach().to(im.dtype), dim=1)

    sim_ii = jdsh_similarity(im, y_for_i, params)
    sim_tt = jdsh_similarity(x_for_t, tx, params)

    c_i = im.mm(x_for_t.t()) * 2.0 - 1.0
    c_t = y_for_i.mm(tx.t()) * 2.0 - 1.0
    s_ii_raw = im.mm(im.t()) * 2.0 - 1.0
    s_tt_raw = tx.mm(tx.t()) * 2.0 - 1.0

    a_i = F.normalize(F.normalize(s_ii_raw, dim=1) * r_row, dim=1)
    a_t = F.normalize(F.normalize(s_tt_raw, dim=1) * r_row, dim=1)
    a_i_rev = F.normalize(F.normalize(c_i.t(), dim=1) * r_row, dim=1)
    a_t_rev = F.normalize(F.normalize(c_t, dim=1) * r_row, dim=1)
    high_it = 0.5 * (a_i.mm(a_t.t()) + a_t_rev.mm(a_i_rev.t()))

    sim_it = _jdsh_stretch(params["alpha"] * c_i + params["beta"] * c_t + params["lamb"] * high_it, params)
    return sim_ii, sim_tt, sim_it


def jdsh_loss_nacg(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    image_codes: torch.Tensor,
    text_codes: torch.Tensor,
    params: Dict[str, Any],
    r_batch: torch.Tensor,
    x_bridge: torch.Tensor,
    y_bridge: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """JDSH-NACG 损失：r ≡ 1 时严格逐位退化回 jdsh_loss；含噪时全保留单模态结构并重构跨模态目标。"""
    if bool(torch.all(r_batch == 1.0).item()):
        return jdsh_loss(image_features, texts, image_codes, text_codes, params)

    sim_ii, sim_tt, sim_it = jdsh_similarity_nacg(
        image_features=image_features,
        texts=texts,
        params=params,
        r_batch=r_batch,
        x_bridge=x_bridge,
        y_bridge=y_bridge,
    )
    ib = F.normalize(image_codes, dim=1)
    tb = F.normalize(text_codes, dim=1)
    r_flat = r_batch.detach().to(dtype=ib.dtype, device=ib.device).view(-1)

    loss_image = F.mse_loss(ib.mm(ib.t()), sim_ii)
    loss_text = F.mse_loss(tb.mm(tb.t()), sim_tt)
    diag_pull = (r_flat * (ib * tb).sum(dim=1)).sum() / r_flat.sum().clamp_min(1e-6)
    loss_cross = F.mse_loss(ib.mm(tb.t()), sim_it) + F.mse_loss(tb.mm(ib.t()), sim_it.t()) - diag_pull
    total = params["intra"] * loss_image + loss_cross + params["intra"] * loss_text
    return total, {"image": loss_image, "cross": loss_cross, "text": loss_text, "sim_it": sim_it}


def djsrh_loss_nacg(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    image_codes: torch.Tensor,
    text_codes: torch.Tensor,
    params: Dict[str, Any],
    r_batch: torch.Tensor,
    x_bridge: torch.Tensor,
    y_bridge: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DJSRH-NACG 损失：r ≡ 1 时严格逐位退化回 djsrh_loss；含噪时单模态免误杀并用净桥重构跨模态目标。"""
    if bool(torch.all(r_batch == 1.0).item()):
        return djsrh_loss(image_features, texts, image_codes, text_codes, params)

    im = F.normalize(image_features)
    tx = F.normalize(texts)
    r_flat = r_batch.detach().to(dtype=im.dtype, device=im.device).view(-1)
    r_col = r_flat.unsqueeze(1)
    b_size = float(texts.shape[0])
    beta = float(params["beta"])
    eta_a = float(params["affinity_eta"])
    mu = float(params["mu"])

    y_for_i = F.normalize(r_col * tx + (1.0 - r_col) * y_bridge.detach().to(im.dtype), dim=1)
    x_for_t = F.normalize(r_col * im + (1.0 - r_col) * x_bridge.detach().to(im.dtype), dim=1)

    si_raw = im.mm(im.t()) * 2.0 - 1.0
    st_raw = tx.mm(tx.t()) * 2.0 - 1.0

    # 无污染 I-I 与 T-T 相似矩阵目标
    st_for_i = y_for_i.mm(y_for_i.t()) * 2.0 - 1.0
    base_ii = beta * si_raw + (1.0 - beta) * st_for_i
    sim_ii = ((1.0 - eta_a) * base_ii + eta_a * base_ii.mm(base_ii) / b_size) * mu

    si_for_t = x_for_t.mm(x_for_t.t()) * 2.0 - 1.0
    base_tt = beta * si_for_t + (1.0 - beta) * st_raw
    sim_tt = ((1.0 - eta_a) * base_tt + eta_a * base_tt.mm(base_tt) / b_size) * mu

    # 非对称跨模态高阶重构目标 S_{IT}^{rect}
    c_i = im.mm(x_for_t.t()) * 2.0 - 1.0
    c_t = y_for_i.mm(tx.t()) * 2.0 - 1.0
    base_it = beta * c_i + (1.0 - beta) * c_t
    sim_ii_pre = (1.0 - eta_a) * base_ii + eta_a * base_ii.mm(base_ii) / b_size
    sim_tt_pre = (1.0 - eta_a) * base_tt + eta_a * base_tt.mm(base_tt) / b_size
    sim_it = ((1.0 - eta_a) * base_it + eta_a * 0.5 * (sim_ii_pre.mm(base_it) + base_it.mm(sim_tt_pre)) / b_size) * mu

    ib = F.normalize(image_codes)
    tb = F.normalize(text_codes)
    li = F.mse_loss(ib.mm(ib.t()), sim_ii)
    lc = F.mse_loss(ib.mm(tb.t()), sim_it)
    lt = F.mse_loss(tb.mm(tb.t()), sim_tt)
    return params["lambda1"] * li + lc + params["lambda2"] * lt, {"image": li, "cross": lc, "text": lt}


class NACGNCEAverage(nn.Module):
    """干净锚点记忆重合成 NCEAverage 包装器。

    当 ``r_batch is None`` 或 ``torch.all(r_batch == 1.0)`` 时，严格委托原生 ``base_nce``；
    激活时用干净桥接特征合成无污染正键写入标量恒定动量记忆槽，并在文本分支替换干净正键锚点。
    """

    def __init__(self, base_nce: Any) -> None:
        super().__init__()
        self.base_nce = base_nce

    @property
    def memory(self) -> torch.Tensor:
        return self.base_nce.memory

    @property
    def params(self) -> torch.Tensor:
        return self.base_nce.params

    @property
    def multinomial(self) -> Any:
        return self.base_nce.multinomial

    def forward(
        self,
        l: torch.Tensor,
        ab: torch.Tensor,
        y: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        r_batch: Optional[torch.Tensor] = None,
        hx_bridge: Optional[torch.Tensor] = None,
        hy_bridge: Optional[torch.Tensor] = None,
        anchor_idx: Optional[torch.Tensor] = None,
        nbr_idx_t: Optional[torch.Tensor] = None,
        nbr_w_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if r_batch is None or bool(torch.all(r_batch == 1.0).item()):
            return self.base_nce(l, ab, y, idx=idx, epoch=epoch)

        assert hx_bridge is not None and hy_bridge is not None
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

        batch_size = int(l.size(0))
        output_size = int(base.memory.size(0))
        input_size = int(base.memory.size(1))
        y_flat = y.view(-1)

        if idx is None:
            drawn = base.multinomial.draw(batch_size * (base.K + 1))
            sample_idx: torch.Tensor = drawn.view(batch_size, -1)
            sample_idx.select(1, 0).copy_(y_flat.data)
        else:
            sample_idx = idx

        r_col = r_batch.detach().view(batch_size, 1).to(dtype=l.dtype, device=l.device)
        hy_for_x = F.normalize(r_col * ab.detach() + (1.0 - r_col) * hy_bridge.detach().to(l.dtype), dim=1)
        v_mem = F.normalize(0.5 * (l.detach() + hy_for_x), dim=1)

        if momentum <= 0.0:
            inx = torch.stack([torch.arange(batch_size)] * batch_size)
            inx = (
                torch.cat(
                    [
                        torch.arange(batch_size).view([-1, 1]),
                        inx[torch.eye(batch_size) == 0].view([batch_size, -1]),
                    ],
                    dim=1,
                )
                .to(l.device)
                .view([-1])
            )
            weight_l = v_mem[inx].view([batch_size, batch_size, input_size]).clone().sign_()

            hx_for_y = F.normalize(r_col * l.detach() + (1.0 - r_col) * hx_bridge.detach().to(ab.dtype), dim=1)
            v_mem_t = F.normalize(0.5 * (hx_for_y + ab.detach()), dim=1)
            weight_ab = weight_l.clone()
            weight_ab[:, 0, :] = v_mem_t.sign()
        else:
            mem_raw = (
                torch.index_select(base.memory, 0, sample_idx.view(-1))
                .detach()
                .view(batch_size, K + 1, input_size)
            )
            weight_l = mem_raw.clone().sign_()
            weight_ab = weight_l.clone()
            if anchor_idx is not None and nbr_idx_t is not None and nbr_w_t is not None:
                mem_anc = torch.index_select(base.memory, 0, anchor_idx).detach()
                mem_bridge_t = gather_bridged_features(mem_anc, nbr_idx_t, nbr_w_t)
                mem_pos_t = F.normalize(
                    r_col * mem_raw[:, 0, :] + (1.0 - r_col) * mem_bridge_t.to(mem_raw.dtype),
                    dim=1,
                )
                weight_ab[:, 0, :] = mem_pos_t.sign()

        out_ab = torch.bmm(weight_ab, ab.view(batch_size, input_size, 1))
        out_l = torch.bmm(weight_l, l.view(batch_size, input_size, 1))

        if base.use_softmax:
            out_ab = torch.div(out_ab, T).contiguous()
            out_l = torch.div(out_l, T).contiguous()
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

        # 恒定标量动量更新：从第 0 轮起即写入干净对齐向量 v_mem，杜绝 Epoch-0 垃圾锁定
        with torch.no_grad():
            l_pos = torch.index_select(base.memory, 0, y_flat)
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v_mem, 1.0 - momentum))
            l_pos = F.normalize(l_pos, dim=1, eps=1e-12)
            base.memory.index_copy_(0, y_flat, l_pos)

        return out_l, out_ab


def compute_clean_cross_sim(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    r_batch: torch.Tensor,
    x_bridge: torch.Tensor,
    y_bridge: torch.Tensor,
) -> torch.Tensor:
    """计算用于假阴性屏蔽与正相关结构正则的干净跨模态相似矩阵 sim_it in [-1, 1]。"""
    im = F.normalize(image_features.to(torch.float32), dim=1, eps=1e-8)
    tx = F.normalize(texts.to(torch.float32), dim=1, eps=1e-8)
    r_col = r_batch.detach().to(dtype=im.dtype, device=im.device).view(-1, 1)
    y_for_i = F.normalize(r_col * tx + (1.0 - r_col) * y_bridge.detach().to(im.dtype), dim=1, eps=1e-8)
    x_for_t = F.normalize(r_col * im + (1.0 - r_col) * x_bridge.detach().to(im.dtype), dim=1, eps=1e-8)
    c_i = im.mm(x_for_t.t()) * 2.0 - 1.0
    c_t = y_for_i.mm(tx.t()) * 2.0 - 1.0
    return (0.5 * (c_i + c_t)).detach()


def nacg_ucch_contrastive_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    r_batch: torch.Tensor,
    hx_bridge: torch.Tensor,
    hy_bridge: torch.Tensor,
    margin: float = 0.2,
    shift: float = 0.1,
    tau: float = 1.0,
    false_neg_sim: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """支持干净桥接正锚替换与高置信多标签同义负样本豁免的 UCCH ContrastiveLoss。"""
    scores = im.mm(s.t())
    r_col = r_batch.detach().to(dtype=im.dtype, device=im.device).view(-1, 1)
    s_pos = r_col * s + (1.0 - r_col) * hy_bridge.detach().to(im.dtype)
    im_pos = r_col * im + (1.0 - r_col) * hx_bridge.detach().to(s.dtype)

    diag_row = (im * s_pos).sum(dim=1, keepdim=True)
    diag_col = (im_pos * s).sum(dim=1, keepdim=True)
    d1 = diag_row.expand_as(scores)
    d2 = diag_col.t().expand_as(scores)

    mask_s = (scores >= (d1 - margin)).float().detach()
    mask_im = (scores >= (d2 - margin)).float().detach()
    if false_neg_sim is not None:
        eye = torch.eye(int(scores.shape[0]), device=scores.device, dtype=torch.bool)
        fn_mask = (false_neg_sim.detach() > 0.25) & (~eye)
        mask_s = mask_s.masked_fill(fn_mask, 0.0)
        mask_im = mask_im.masked_fill(fn_mask, 0.0)

    cost_s = scores * mask_s + (1.0 - mask_s) * (scores - shift)
    cost_im = scores * mask_im + (1.0 - mask_im) * (scores - shift)

    row_terms = -diag_row.view(-1) + float(tau) * (cost_s / float(tau)).exp().sum(1).log() + float(margin)
    col_terms = -diag_col.view(-1) + float(tau) * (cost_im / float(tau)).exp().sum(0).log() + float(margin)
    return row_terms.mean() + col_terms.mean()


def nacg_positive_graph_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    sim_it: torch.Tensor,
    sim_ii: Optional[torch.Tensor] = None,
    sim_tt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """仅对高置信正相关近邻 (sim > 0) 施加对称图一致性辅助损失，避免负排斥损伤弱多标签集合。"""
    def _masked_pos_mse(pred: torch.Tensor, target_raw: torch.Tensor) -> torch.Tensor:
        pos_mask = target_raw > 0.0
        target = target_raw.clamp(0.0, 1.0).to(dtype=pred.dtype)
        diff_sq = (pred - target).square()
        masked = torch.where(pos_mask, diff_sq, torch.zeros_like(diff_sq))
        return masked.sum() / pos_mask.to(pred.dtype).sum().clamp_min(1.0)

    sym_it = 0.5 * (sim_it + sim_it.t())
    t_ii = sym_it if sim_ii is None else sim_ii
    t_tt = sym_it if sim_tt is None else sim_tt

    li = _masked_pos_mse(im.mm(im.t()), t_ii)
    lt = _masked_pos_mse(s.mm(s.t()), t_tt)
    l_cross = 0.5 * (_masked_pos_mse(im.mm(s.t()), sim_it) + _masked_pos_mse(s.mm(im.t()), sim_it.t()))
    return (li + lt + l_cross) / 3.0


def compute_rch_sample_weights(
    r_batch: torch.Tensor,
    gamma_r: float = 2.0,
    delta: float = 0.02,
) -> torch.Tensor:
    """根据干净后验可信度计算 RCH-v2 逐样本权重 w_i = (1 - delta) * r_i^{gamma_r} + delta。"""
    if bool(torch.all(r_batch == 1.0).item()):
        return torch.ones_like(r_batch, dtype=torch.float32).detach()
    r_det = r_batch.detach().to(torch.float32).view(-1).clamp(0.0, 1.0)
    return ((1.0 - float(delta)) * r_det.pow(float(gamma_r)) + float(delta)).detach()


class RCHNCEAverage(nn.Module):
    """无锚点均值模糊的干净门控记忆库 NCE 包装器（RCH-v2）。

    当 ``r_batch is None`` 或 ``torch.all(r_batch == 1.0)`` 时严格委托原生 ``base_nce``；
    含噪时使用 r_i^{gamma_r} 抑制错配文本进入实例记忆目标 v_mem = normalize(l + r_pow * ab)，
    始终以清晰个体视觉编码而非多锚点均匀均值保持记忆键锐度与辨别力。
    """

    def __init__(self, base_nce: Any) -> None:
        super().__init__()
        self.base_nce = base_nce

    @property
    def memory(self) -> torch.Tensor:
        return self.base_nce.memory

    @property
    def params(self) -> torch.Tensor:
        return self.base_nce.params

    @property
    def multinomial(self) -> Any:
        return self.base_nce.multinomial

    def forward(
        self,
        l: torch.Tensor,
        ab: torch.Tensor,
        y: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        r_batch: Optional[torch.Tensor] = None,
        gamma_r: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if r_batch is None or bool(torch.all(r_batch == 1.0).item()):
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

        batch_size = int(l.size(0))
        output_size = int(base.memory.size(0))
        input_size = int(base.memory.size(1))
        y_flat = y.view(-1)

        if idx is None:
            drawn = base.multinomial.draw(batch_size * (base.K + 1))
            sample_idx: torch.Tensor = drawn.view(batch_size, -1)
            sample_idx.select(1, 0).copy_(y_flat.data)
        else:
            sample_idx = idx

        r_pow = (
            r_batch.detach()
            .view(batch_size, 1)
            .to(dtype=l.dtype, device=l.device)
            .clamp(0.0, 1.0)
            .pow(float(gamma_r))
        )
        v_mem = F.normalize(l.detach() + r_pow * ab.detach(), dim=1, eps=1e-12)

        if momentum <= 0.0:
            inx = torch.stack([torch.arange(batch_size)] * batch_size)
            inx = (
                torch.cat(
                    [
                        torch.arange(batch_size).view([-1, 1]),
                        inx[torch.eye(batch_size) == 0].view([batch_size, -1]),
                    ],
                    dim=1,
                )
                .to(l.device)
                .view([-1])
            )
            weight = v_mem[inx].view([batch_size, batch_size, input_size]).clone().sign_()
        else:
            weight = (
                torch.index_select(base.memory, 0, sample_idx.view(-1))
                .detach()
                .view(batch_size, K + 1, input_size)
                .clone()
                .sign_()
            )

        out_ab = torch.bmm(weight, ab.view(batch_size, input_size, 1))
        out_l = torch.bmm(weight, l.view(batch_size, input_size, 1))

        if base.use_softmax:
            out_ab = torch.div(out_ab, T).contiguous()
            out_l = torch.div(out_l, T).contiguous()
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
            l_pos = torch.index_select(base.memory, 0, y_flat)
            l_pos.mul_(momentum).add_(torch.mul(v_mem, 1.0 - momentum))
            l_pos = F.normalize(l_pos, dim=1, eps=1e-12)
            base.memory.index_copy_(0, y_flat, l_pos)

        return out_l, out_ab


def compute_centered_clean_graph(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    r_batch: torch.Tensor,
    valid_batch: torch.Tensor,
    state: NACGAnchorState,
    high_order_weight: float = 0.30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """仅基于训练集中心化均值/IDF 统计构造成对干净图 (g_ii, g_tt, g_it) 与可信度矩阵。"""
    assert state.mu_x is not None and state.mu_y is not None, "NACGAnchorState 缺少中心化统计 mu_x/mu_y"
    with torch.no_grad():
        x_f = image_features.to(torch.float32)
        y_f = texts.to(torch.float32)
        dev = x_f.device
        v_f = valid_batch.to(device=dev, dtype=torch.float32).view(-1)

        xc = F.normalize(x_f - state.mu_x.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8)
        y_proc = torch.log1p(y_f.clamp_min(0.0)) * state.idf_y.to(device=dev, dtype=torch.float32) if state.idf_y is not None else y_f
        yc = F.normalize(y_proc - state.mu_y.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8) * v_f.unsqueeze(1)

        s_ii = xc.mm(xc.t())
        s_tt = yc.mm(yc.t())

        u = r_batch.detach().to(device=dev, dtype=torch.float32).view(-1).clamp(0.0, 1.0) * v_f
        uu = u.unsqueeze(1) * u.unsqueeze(0)
        vv = v_f.unsqueeze(1) * v_f.unsqueeze(0)

        a_ii = (s_ii + uu * s_tt) / (1.0 + uu)
        a_tt = vv * (s_tt + uu * s_ii) / (1.0 + uu)

        ho = float(high_order_weight)
        p_ii = F.normalize(a_ii, dim=1, eps=1e-8)
        p_tt = F.normalize(a_tt, dim=1, eps=1e-8) * v_f.unsqueeze(1)
        g_ii = (1.0 - ho) * a_ii + ho * p_ii.mm(p_ii.t())
        g_tt = vv * ((1.0 - ho) * a_tt + ho * p_tt.mm(p_tt.t()))

        u_row = u.unsqueeze(0)
        u_col = u.unsqueeze(1) * v_f.unsqueeze(0)
        direct_it = (u_row * g_ii + u_col * g_tt) / (u_row + u_col).clamp_min(1e-6)
        p_i_clean = F.normalize(g_ii * u_row, dim=1, eps=1e-8)
        p_t_clean = F.normalize(g_tt * u_row, dim=1, eps=1e-8) * v_f.unsqueeze(1)
        bridge_it = p_i_clean.mm(p_t_clean.t())
        has_direct = ((u_row + u_col) > 0.15).to(torch.float32)
        g_it = has_direct * ((1.0 - ho) * direct_it + ho * bridge_it) + (1.0 - has_direct) * bridge_it

        omega_ii = torch.ones_like(g_ii)
        omega_tt = vv
        omega_it = v_f.unsqueeze(0) * (1.0 - (1.0 - ho) * (1.0 - u.unsqueeze(1)) * (1.0 - u.unsqueeze(0)))
        return (
            g_ii.detach(),
            g_tt.detach(),
            g_it.detach(),
            omega_ii.detach(),
            omega_tt.detach(),
            omega_it.detach(),
        )


def rch_contrastive_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    w_batch: torch.Tensor,
    margin: float = 0.2,
    shift: float = 0.1,
    tau: float = 1.0,
    false_neg_sim: Optional[torch.Tensor] = None,
    false_neg_conf: Optional[torch.Tensor] = None,
    pos_quantile: float = 0.85,
) -> torch.Tensor:
    """无均值坍缩的正向可信度加权 ContrastiveLoss（支持自适应分位数假阴性屏蔽）。"""
    scores = im.mm(s.t())
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask_s = (scores >= (d1 - float(margin))).float().detach()
    mask_im = (scores >= (d2 - float(margin))).float().detach()
    if false_neg_sim is not None:
        eye = torch.eye(int(scores.shape[0]), device=scores.device, dtype=torch.bool)
        off_sim = false_neg_sim.detach().masked_select(~eye)
        if off_sim.numel() > 0:
            tau_fn = torch.quantile(off_sim, float(pos_quantile)).clamp(min=0.12, max=0.35)
        else:
            tau_fn = 0.20
        conf_ok = (false_neg_conf.detach() > 0.40) if false_neg_conf is not None else torch.ones_like(eye)
        fn_mask = (~eye) & conf_ok & (false_neg_sim.detach() > tau_fn)
        mask_s = mask_s.masked_fill(fn_mask, 0.0)
        mask_im = mask_im.masked_fill(fn_mask, 0.0)

    cost_s = scores * mask_s + (1.0 - mask_s) * (scores - float(shift))
    cost_im = scores * mask_im + (1.0 - mask_im) * (scores - float(shift))

    row_terms = -cost_s.diag() + float(tau) * (cost_s / float(tau)).exp().sum(1).log() + float(margin)
    col_terms = -cost_im.diag() + float(tau) * (cost_im / float(tau)).exp().sum(0).log() + float(margin)

    w_flat = w_batch.detach().view(-1).to(dtype=row_terms.dtype, device=row_terms.device)
    denom = w_flat.sum().clamp_min(1e-12)
    return (w_flat * row_terms).sum() / denom + (w_flat * col_terms).sum() / denom


def rch_clean_graph_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    g_ii: torch.Tensor,
    g_tt: torch.Tensor,
    g_it: torch.Tensor,
    omega_ii: torch.Tensor,
    omega_tt: torch.Tensor,
    omega_it: torch.Tensor,
    pos_quantile: float = 0.85,
    stretch: float = 1.8,
) -> torch.Tensor:
    """自适应分位数正向干净邻域图一致性损失。"""
    eye = torch.eye(int(im.shape[0]), device=im.device, dtype=torch.bool)

    def _pos_term(pred: torch.Tensor, g_mat: torch.Tensor, omega_mat: torch.Tensor) -> torch.Tensor:
        valid_off = (~eye) & (omega_mat > 0.25)
        if int(valid_off.sum().item()) < 8:
            return pred.sum() * 0.0
        vals = g_mat.masked_select(valid_off)
        thr = torch.quantile(vals, float(pos_quantile)).clamp(min=0.10, max=0.38)
        pos_mask = valid_off & (g_mat > thr)
        w = torch.where(pos_mask, omega_mat.to(pred.dtype), torch.zeros_like(pred))
        w_sum = w.sum()
        if float(w_sum.item()) <= 1e-8:
            return pred.sum() * 0.0
        target = (float(stretch) * g_mat).clamp(0.0, 1.0).to(pred.dtype)
        return (w * (pred - target).square()).sum() / w_sum.clamp_min(1.0)

    li = _pos_term(im.mm(im.t()), g_ii, omega_ii)
    lt = _pos_term(s.mm(s.t()), g_tt, omega_tt)
    l_it = 0.5 * (_pos_term(im.mm(s.t()), g_it, omega_it) + _pos_term(s.mm(im.t()), g_it.t(), omega_it.t()))
    return (li + lt + l_it) / 3.0


def compute_zscore_clean_graph(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    r_batch: torch.Tensor,
    valid_batch: torch.Tensor,
    state: NACGAnchorState,
    high_order_weight: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建非对角 Z-score 标准化的一阶+二阶正轮廓干净多模态亲和图（修复稀疏 BoW 均值稀释与二跳置信截断）。

    [P2-2] 稀疏 BoW 约定：
      - 当 ``state.is_sparse_bow=True`` 时，直接对非负 ``log1p(y_f) * idf_y`` 做单位归一化后计算余弦内积，
        由后续 ``_offdiag_zscore`` 在成对相似度矩阵层面完成零均值单位方差标准化，避免稠密 ``mu_y`` 稀释短词袋；
      - 当 ``state.is_sparse_bow=False``（稠密连续文本表示）时，沿用 ``y_proc - mu_y`` 中心化余弦核。
    [P1-3] 二跳纯桥接置信度自适应提升：
      - ``ho_eff = clip(ho + 0.25 * sigmoid(z_bridge_it - 1.0), 0.0, 1.0)``（默认 ``ho=0.35``），
        使高显著度二跳纯桥接（``z_bridge_it > 1.35``）获得 ``omega_it > 0.42``，顺利通过下游 ``0.35 / 0.40`` 门限。
    """
    assert state.mu_x is not None and state.mu_y is not None, "NACGAnchorState 缺少中心化统计 mu_x/mu_y"
    with torch.no_grad():
        x_f = image_features.to(torch.float32)
        y_f = texts.to(torch.float32)
        dev = x_f.device
        b_size = int(x_f.shape[0])
        eye = torch.eye(b_size, device=dev, dtype=torch.bool)
        v_f = valid_batch.to(device=dev, dtype=torch.float32).view(-1)
        v_bool = v_f > 0.5
        vv_bool = v_bool.unsqueeze(1) & v_bool.unsqueeze(0)
        ones_bool = torch.ones((b_size, b_size), device=dev, dtype=torch.bool)
        it_valid_bool = ones_bool & v_bool.unsqueeze(0)

        xc = F.normalize(x_f - state.mu_x.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8)
        if state.is_sparse_bow and state.idf_y is not None:
            y_tfidf = torch.log1p(y_f.clamp_min(0.0)) * state.idf_y.to(device=dev, dtype=torch.float32)
            yc = F.normalize(y_tfidf, dim=1, eps=1e-8) * v_f.unsqueeze(1)
        else:
            y_proc = (
                torch.log1p(y_f.clamp_min(0.0)) * state.idf_y.to(device=dev, dtype=torch.float32)
                if state.idf_y is not None
                else y_f
            )
            yc = F.normalize(y_proc - state.mu_y.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8) * v_f.unsqueeze(1)

        def _offdiag_zscore(mat: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
            sel = (~eye) & pair_mask
            if int(sel.sum().item()) < 8:
                return torch.zeros_like(mat)
            vals = mat.masked_select(sel)
            mu = vals.mean()
            sd = vals.std(unbiased=False).clamp_min(1e-5)
            z = ((mat - mu) / sd).clamp(-4.0, 6.0)
            return torch.where(pair_mask, z, torch.zeros_like(z))

        s_ii = xc.mm(xc.t())
        s_tt = yc.mm(yc.t())
        z_ii_1 = _offdiag_zscore(s_ii, ones_bool)
        z_tt_1 = _offdiag_zscore(s_tt, vv_bool)

        u = r_batch.detach().to(device=dev, dtype=torch.float32).view(-1).clamp(0.0, 1.0) * v_f
        uu = (u.unsqueeze(1) * u.unsqueeze(0)).clamp(0.0, 1.0)
        vv = v_f.unsqueeze(1) * v_f.unsqueeze(0)

        a_ii = (z_ii_1 + uu * z_tt_1) / torch.sqrt(1.0 + uu.square())
        a_tt = vv * (z_tt_1 + uu * z_ii_1) / torch.sqrt(1.0 + uu.square())

        ho = float(high_order_weight)
        p_ii = F.normalize(F.relu(a_ii).masked_fill(eye, 0.0), dim=1, eps=1e-8)
        p_tt = F.normalize(F.relu(a_tt).masked_fill(eye, 0.0), dim=1, eps=1e-8) * v_f.unsqueeze(1)
        z_ho_ii = _offdiag_zscore(p_ii.mm(p_ii.t()), ones_bool)
        z_ho_tt = _offdiag_zscore(p_tt.mm(p_tt.t()), vv_bool)
        z_ii = _offdiag_zscore((1.0 - ho) * a_ii + ho * z_ho_ii, ones_bool)
        z_tt = _offdiag_zscore((1.0 - ho) * a_tt + ho * z_ho_tt, vv_bool)

        u_row = u.unsqueeze(0)
        u_col = u.unsqueeze(1) * v_f.unsqueeze(0)
        direct_it = (u_row * z_ii + u_col * z_tt) / torch.sqrt(u_row.square() + u_col.square()).clamp_min(1e-6)
        p_i_clean = F.normalize(F.relu(z_ii).masked_fill(eye, 0.0) * u_row, dim=1, eps=1e-8)
        p_t_clean = F.normalize(F.relu(z_tt).masked_fill(eye, 0.0) * u_row, dim=1, eps=1e-8) * v_f.unsqueeze(1)
        z_bridge_it = _offdiag_zscore(p_i_clean.mm(p_t_clean.t()), it_valid_bool)
        has_direct = ((u_row + u_col) > 0.20).to(torch.float32)
        raw_it = has_direct * ((1.0 - ho) * direct_it + ho * z_bridge_it) + (1.0 - has_direct) * z_bridge_it
        z_it = _offdiag_zscore(raw_it, it_valid_bool)

        ho_eff = (ho + 0.25 * torch.sigmoid(z_bridge_it - 1.0)).clamp(0.0, 1.0)
        omega_ii = torch.ones_like(z_ii)
        omega_tt = vv
        omega_it = v_f.unsqueeze(0) * (1.0 - (1.0 - ho_eff) * (1.0 - u.unsqueeze(1)) * (1.0 - u.unsqueeze(0)))
        return (
            z_ii.detach(),
            z_tt.detach(),
            z_it.detach(),
            omega_ii.detach(),
            omega_tt.detach(),
            omega_it.detach(),
        )


class RCHv3NCEAverage(nn.Module):
    """模态解耦干净孪生记忆刷新 NCE 包装器（RCH-v3，已修复 [P1-2] 文本侧原型碰撞）。

    预热轮（momentum <= 0）或 r ≡ 1 时严格走原生对称 NCEAverage；
    后续轮次中将 ``weight_l[:, 0, :]`` 与 ``weight_ab[:, 0, :]`` 解耦：
      - 视觉分支对低可信样本路由至高置信干净视觉孪生 ``z_ii`` 原型；
      - 文本分支若有 ``z_tt`` 干净文本孪生则路由至该文本孪生原型，否则保持未受视觉孪生覆盖污染的原始记忆槽，
        绝不把错配文本强拉向视觉孪生原型。
    """

    def __init__(self, base_nce: Any) -> None:
        super().__init__()
        self.base_nce = base_nce
        self._text_safe_mem: Optional[torch.Tensor] = None

    @property
    def memory(self) -> torch.Tensor:
        return self.base_nce.memory

    @property
    def params(self) -> torch.Tensor:
        return self.base_nce.params

    @property
    def multinomial(self) -> Any:
        return self.base_nce.multinomial

    def forward(
        self,
        l: torch.Tensor,
        ab: torch.Tensor,
        y: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        r_batch: Optional[torch.Tensor] = None,
        z_ii: Optional[torch.Tensor] = None,
        valid_batch: Optional[torch.Tensor] = None,
        z_tt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        momentum = (
            float(self.base_nce.params[4].item())
            if (epoch is None)
            else (0.0 if epoch < 0 else float(self.base_nce.params[4].item()))
        )
        if r_batch is None or momentum <= 0.0 or bool(torch.all(r_batch == 1.0).item()):
            out_l_base, out_ab_base = self.base_nce(l, ab, y, idx=idx, epoch=epoch)
            if self._text_safe_mem is not None:
                self._text_safe_mem = self.base_nce.memory.detach().clone()
            return out_l_base, out_ab_base

        base = self.base_nce
        if self._text_safe_mem is None or self._text_safe_mem.shape != base.memory.shape or self._text_safe_mem.device != base.memory.device:
            self._text_safe_mem = base.memory.detach().clone()

        K = int(base.params[0].item())
        T = float(base.params[1].item())
        Z_l = float(base.params[2].item())
        Z_ab = float(base.params[3].item())

        batch_size = int(l.size(0))
        output_size = int(base.memory.size(0))
        input_size = int(base.memory.size(1))
        y_flat = y.view(-1)

        if idx is None:
            drawn = base.multinomial.draw(batch_size * (base.K + 1))
            sample_idx: torch.Tensor = drawn.view(batch_size, -1)
            sample_idx.select(1, 0).copy_(y_flat.data)
        else:
            sample_idx = idx

        r_flat = r_batch.detach().view(batch_size).to(dtype=l.dtype, device=l.device).clamp(0.0, 1.0)
        v_bool = (
            (valid_batch.detach().view(batch_size).to(device=l.device) > 0.5)
            if valid_batch is not None
            else torch.ones(batch_size, device=l.device, dtype=torch.bool)
        )
        v_f = v_bool.to(dtype=l.dtype)
        low_r = r_flat < 0.55

        mem_raw = (
            torch.index_select(base.memory, 0, sample_idx.view(-1))
            .detach()
            .view(batch_size, K + 1, input_size)
        )
        weight_l = mem_raw.clone().sign_()
        weight_ab = mem_raw.clone().sign_()

        # 文本正锚初始取未受视觉孪生覆盖的记忆槽，阻断跨轮次视觉覆盖导致的模态碰撞
        if bool(low_r.any().item()):
            low_idx = torch.nonzero(low_r, as_tuple=False).view(-1)
            orig_pos_t = torch.index_select(self._text_safe_mem, 0, y_flat.index_select(0, low_idx)).detach().to(weight_ab.dtype).sign()
            weight_ab[low_idx, 0, :] = orig_pos_t

        has_twin_i = torch.zeros(batch_size, device=l.device, dtype=torch.bool)
        best_j_i = torch.arange(batch_size, device=l.device, dtype=torch.long)
        if z_ii is not None and batch_size >= 2:
            eye = torch.eye(batch_size, device=l.device, dtype=torch.bool)
            clean_i_cand = (r_flat > 0.65).unsqueeze(0) & (~eye)
            twin_scores_i = (z_ii.detach().to(dtype=l.dtype, device=l.device) * r_flat.unsqueeze(0)).masked_fill(~clean_i_cand, -1e9)
            best_val_i, best_j_i = twin_scores_i.max(dim=1)
            has_twin_i = low_r & (best_val_i > 1.2)
            if bool(has_twin_i.any().item()):
                src_i = torch.nonzero(has_twin_i, as_tuple=False).view(-1)
                twin_mem_i = torch.index_select(base.memory, 0, y_flat.index_select(0, best_j_i.index_select(0, src_i))).detach().to(weight_l.dtype).sign()
                weight_l[src_i, 0, :] = twin_mem_i

        if z_tt is not None and batch_size >= 2:
            eye = torch.eye(batch_size, device=l.device, dtype=torch.bool)
            clean_t_cand = ((r_flat > 0.65) & v_bool).unsqueeze(0) & (~eye)
            u_j_t = (r_flat * v_f).unsqueeze(0)
            twin_scores_t = (z_tt.detach().to(dtype=l.dtype, device=l.device) * u_j_t).masked_fill(~clean_t_cand, -1e9)
            best_val_t, best_j_t = twin_scores_t.max(dim=1)
            has_twin_t = low_r & v_bool & (best_val_t > 1.2)
            if bool(has_twin_t.any().item()):
                src_t = torch.nonzero(has_twin_t, as_tuple=False).view(-1)
                twin_mem_t = torch.index_select(self._text_safe_mem, 0, y_flat.index_select(0, best_j_t.index_select(0, src_t))).detach().to(weight_ab.dtype).sign()
                weight_ab[src_t, 0, :] = twin_mem_t

        out_ab = torch.bmm(weight_ab, ab.view(batch_size, input_size, 1))
        out_l = torch.bmm(weight_l, l.view(batch_size, input_size, 1))

        if base.use_softmax:
            out_ab = torch.div(out_ab, T).contiguous()
            out_l = torch.div(out_l, T).contiguous()
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
            r_col = r_flat.view(batch_size, 1)
            v_orig = F.normalize(0.5 * (l.detach() + ab.detach()), dim=1, eps=1e-12)

            # _text_safe_mem 仅按干净可信度更新对称特征，永不写入视觉孪生覆盖
            t_pos = torch.index_select(self._text_safe_mem, 0, y_flat)
            om_safe = (1.0 - momentum) * r_col.to(t_pos.dtype)
            t_pos.mul_(1.0 - om_safe).add_(torch.mul(v_orig.to(t_pos.dtype), om_safe))
            t_pos = F.normalize(t_pos, dim=1, eps=1e-12)
            self._text_safe_mem.index_copy_(0, y_flat, t_pos)

            if z_ii is not None and valid_batch is not None and batch_size >= 4:
                has_t_col = has_twin_i.to(dtype=l.dtype).unsqueeze(1)
                ab_twin = ab.detach().index_select(0, best_j_i)
                ab_clean = r_col * ab.detach() + (1.0 - r_col) * ab_twin
                v_twin = F.normalize(0.5 * (l.detach() + ab_clean), dim=1, eps=1e-12)
                v_mem = has_t_col * v_twin + (1.0 - has_t_col) * v_orig
                eff_pi = torch.maximum(r_col, has_t_col * 0.70)
            else:
                v_mem = v_orig
                eff_pi = r_col

            l_pos = torch.index_select(base.memory, 0, y_flat)
            one_minus_m = (1.0 - momentum) * eff_pi.to(l_pos.dtype)
            m_col = 1.0 - one_minus_m
            l_pos.mul_(m_col).add_(torch.mul(v_mem.to(l_pos.dtype), one_minus_m))
            l_pos = F.normalize(l_pos, dim=1, eps=1e-12)
            base.memory.index_copy_(0, y_flat, l_pos)

        return out_l, out_ab


class RCHv4NCEAverage(nn.Module):
    """RCH-v4 全解耦双模态干净孪生路由 + Epoch-0 无降权预热/干净门控记忆库 + K 槽假阴性抑制 NCE 包装器。

    三大核心机制：
      1. Epoch 0 预热期（``epoch < 0``）保持 ``w_l = w_ab = 1.0`` 充分激活单模态初始特征学习，
         同时从 Epoch 0 起即对记忆库更新执行 ``pi_mem = r_batch^2`` 干净门控（杜绝噪声嵌合锁入记忆槽）；
      2. 正式轮次（``epoch >= 0``）视觉分支 ``(weight_l[:, 0], w_l)`` 与文本分支 ``(weight_ab[:, 0], w_ab)``
         完全独立按各自单模态干净孪生路由并救援权重至 ``0.80 * r_twin``（无孪生噪声样本保持 ``delta = 0.0`` 零错配强拉）；
      3. 利用冷启动无监督锚点投影 ``(px_norm, py_norm)`` 在 O(B*K*M) 复杂度下检测并屏蔽 K=4096 记忆库中的共现语义假阴性槽位
         （``use_softmax=True`` 时对负例预指数 logit 置 ``-1e9`` 使 softmax 排斥精确为 0；``use_softmax=False`` 时对负例概率置 ``0.0``
         使 ``log(1 - P_neg) == 0``，全程零 NaN）。
    """

    def __init__(self, base_nce: Any) -> None:
        super().__init__()
        self.base = base_nce
        self.base_nce = base_nce

    @property
    def memory(self) -> torch.Tensor:
        return self.base_nce.memory

    @property
    def params(self) -> torch.Tensor:
        return self.base_nce.params

    @property
    def multinomial(self) -> Any:
        return self.base_nce.multinomial

    def forward(
        self,
        l: torch.Tensor,
        ab: torch.Tensor,
        y: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        epoch: Optional[int] = 0,
        r_all: Optional[torch.Tensor] = None,
        px_norm: Optional[torch.Tensor] = None,
        py_norm: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        z_ii: Optional[torch.Tensor] = None,
        z_tt: Optional[torch.Tensor] = None,
        z_twin_thr: float = 1.05,
        z_nce_fn_thr: float = 1.00,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.base_nce
        K = int(base.params[0].item())
        T = float(base.params[1].item())
        Z_l = float(base.params[2].item())
        Z_ab = float(base.params[3].item())
        is_warmup = (epoch is not None) and (int(epoch) < 0)
        momentum = 0.0 if is_warmup else float(base.params[4].item())

        batch_size = int(l.size(0))
        output_size = int(base.memory.size(0))
        input_size = int(base.memory.size(1))
        y_flat = y.view(-1)

        if r_all is None:
            r_all_dev = torch.ones(output_size, device=l.device, dtype=torch.float32)
            r_batch = torch.ones(batch_size, device=l.device, dtype=torch.float32)
        else:
            r_det = r_all.detach().view(-1).to(device=l.device, dtype=torch.float32).clamp(0.0, 1.0)
            if int(r_det.numel()) == output_size:
                r_all_dev = r_det
                r_batch = r_all_dev.index_select(0, y_flat)
            elif int(r_det.numel()) == batch_size:
                r_batch = r_det
                r_all_dev = torch.ones(output_size, device=l.device, dtype=torch.float32)
                r_all_dev.index_copy_(0, y_flat, r_batch)
            else:
                raise ValueError(f"r_all 长度 {int(r_det.numel())} 与记忆库大小 {output_size} 或批大小 {batch_size} 不匹配")

        if valid_mask is None:
            valid_all = torch.ones(output_size, device=l.device, dtype=torch.bool)
            valid_b = torch.ones(batch_size, device=l.device, dtype=torch.bool)
        else:
            vm_det = (valid_mask.detach().view(-1) > 0.5).to(device=l.device)
            if int(vm_det.numel()) == output_size:
                valid_all = vm_det
                valid_b = valid_all.index_select(0, y_flat)
            elif int(vm_det.numel()) == batch_size:
                valid_b = vm_det
                valid_all = torch.ones(output_size, device=l.device, dtype=torch.bool)
                valid_all.index_copy_(0, y_flat, valid_b)
            else:
                raise ValueError(f"valid_mask 长度 {int(vm_det.numel())} 与记忆库大小 {output_size} 或批大小 {batch_size} 不匹配")

        if is_warmup:
            w_l = torch.ones_like(r_batch).detach()
            w_ab = torch.ones_like(r_batch).detach()
        else:
            w_l = r_batch.pow(2).clone().detach()
            w_ab = r_batch.pow(2).clone().detach()

        v_sym = F.normalize(0.5 * (l.detach() + ab.detach()), dim=1, eps=1e-12)

        if momentum <= 0.0:
            row_arange = torch.arange(batch_size, device=l.device)
            inx_grid = torch.stack([row_arange] * batch_size)
            eye_b = torch.eye(batch_size, device=l.device, dtype=torch.bool)
            inx_mat = torch.cat([row_arange.view(-1, 1), inx_grid[~eye_b].view(batch_size, -1)], dim=1)
            neg_idx = y_flat.index_select(0, inx_mat[:, 1:].reshape(-1)).view(batch_size, -1)
            weight_base = v_sym.index_select(0, inx_mat.reshape(-1)).view(batch_size, batch_size, input_size)
            weight_l = weight_base.clone().sign_()
            weight_ab = weight_base.clone().sign_()
        else:
            if idx is None:
                drawn = base.multinomial.draw(batch_size * (base.K + 1))
                sample_idx: torch.Tensor = drawn.view(batch_size, -1)
                sample_idx.select(1, 0).copy_(y_flat.data)
            else:
                sample_idx = idx
            neg_idx = sample_idx[:, 1:]
            mem_raw = (
                torch.index_select(base.memory, 0, sample_idx.view(-1))
                .detach()
                .view(batch_size, K + 1, input_size)
            )
            weight_l = mem_raw.clone().sign_()
            weight_ab = mem_raw.clone().sign_()

        # 正式轮次独立双模态干净孪生正锚路由与样本权重救援（预热期保持 w_l = w_ab = 1.0）
        if not is_warmup and z_ii is not None and batch_size >= 2:
            eye = torch.eye(batch_size, device=l.device, dtype=torch.bool)
            clean_cand_i = (r_batch > 0.65).unsqueeze(0) & (~eye)
            z_ii_m = z_ii.detach().to(device=l.device, dtype=torch.float32).masked_fill(~clean_cand_i, -1e9)
            best_z_i, best_j_i = z_ii_m.max(dim=1)
            rescue_i = (r_batch < 0.55) & (best_z_i > float(z_twin_thr))
            if bool(rescue_i.any().item()):
                src_i = torch.nonzero(rescue_i, as_tuple=False).view(-1)
                twin_j_i = best_j_i.index_select(0, src_i)
                if momentum > 0.0:
                    pos_proto_i = base.memory.index_select(0, y_flat.index_select(0, twin_j_i)).detach().to(weight_l.dtype).sign()
                else:
                    pos_proto_i = v_sym.index_select(0, twin_j_i).to(weight_l.dtype).sign()
                weight_l[src_i, 0, :] = pos_proto_i
                w_l[src_i] = torch.maximum(w_l.index_select(0, src_i), 0.80 * r_batch.index_select(0, twin_j_i))

        if not is_warmup and z_tt is not None and batch_size >= 2:
            eye = torch.eye(batch_size, device=l.device, dtype=torch.bool)
            clean_cand_t = ((r_batch > 0.65) & valid_b).unsqueeze(0) & (~eye)
            z_tt_m = z_tt.detach().to(device=l.device, dtype=torch.float32).masked_fill(~clean_cand_t, -1e9)
            best_z_t, best_j_t = z_tt_m.max(dim=1)
            rescue_t = valid_b & (r_batch < 0.55) & (best_z_t > float(z_twin_thr))
            if bool(rescue_t.any().item()):
                src_t = torch.nonzero(rescue_t, as_tuple=False).view(-1)
                twin_j_t = best_j_t.index_select(0, src_t)
                if momentum > 0.0:
                    pos_proto_t = base.memory.index_select(0, y_flat.index_select(0, twin_j_t)).detach().to(weight_ab.dtype).sign()
                else:
                    pos_proto_t = v_sym.index_select(0, twin_j_t).to(weight_ab.dtype).sign()
                weight_ab[src_t, 0, :] = pos_proto_t
                w_ab[src_t] = torch.maximum(w_ab.index_select(0, src_t), 0.80 * r_batch.index_select(0, twin_j_t))

        # K 槽（或预热批内）语义假阴性检测
        n_neg = int(neg_idx.size(1))
        fn_neg_l: Optional[torch.Tensor] = None
        fn_neg_ab: Optional[torch.Tensor] = None
        if n_neg >= 2 and (px_norm is not None or py_norm is not None):
            r_neg = r_all_dev.index_select(0, neg_idx.reshape(-1)).view(batch_size, n_neg)
            r_neg_ok = r_neg > 0.55
            if px_norm is not None and int(px_norm.numel()) > 0:
                px_dev = px_norm.detach().to(device=l.device, dtype=torch.float32)
                q_px = px_dev.index_select(0, y_flat)
                k_px = px_dev.index_select(0, neg_idx.reshape(-1)).view(batch_size, n_neg, -1)
                sim_x = torch.bmm(q_px.unsqueeze(1), k_px.transpose(1, 2)).squeeze(1)
                q_px_ok = q_px.norm(dim=1, keepdim=True) > 1e-6
                mu_x_n = sim_x.mean(dim=1, keepdim=True)
                sd_x_n = sim_x.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-5)
                z_neg_x = (sim_x - mu_x_n) / sd_x_n
                fn_neg_l = q_px_ok & (z_neg_x > float(z_nce_fn_thr)) & r_neg_ok

            if py_norm is not None and int(py_norm.numel()) > 0:
                py_dev = py_norm.detach().to(device=l.device, dtype=torch.float32)
                q_py = py_dev.index_select(0, y_flat)
                k_py = py_dev.index_select(0, neg_idx.reshape(-1)).view(batch_size, n_neg, -1)
                sim_y = torch.bmm(q_py.unsqueeze(1), k_py.transpose(1, 2)).squeeze(1)
                v_neg = valid_all.index_select(0, neg_idx.reshape(-1)).view(batch_size, n_neg)
                v_neg_f = v_neg.to(torch.float32)
                v_cnt = v_neg_f.sum(dim=1, keepdim=True)
                mu_y_n = (sim_y * v_neg_f).sum(dim=1, keepdim=True) / v_cnt.clamp_min(1.0)
                var_y_n = ((sim_y - mu_y_n) * v_neg_f).square().sum(dim=1, keepdim=True) / v_cnt.clamp_min(1.0)
                sd_y_n = torch.sqrt(var_y_n).clamp_min(1e-5)
                z_neg_y = (sim_y - mu_y_n) / sd_y_n
                q_py_ok = valid_b.unsqueeze(1) & (q_py.norm(dim=1, keepdim=True) > 1e-6) & (v_cnt >= 2.0)
                fn_neg_ab = q_py_ok & v_neg & (z_neg_y > float(z_nce_fn_thr)) & r_neg_ok

        out_ab = torch.bmm(weight_ab, ab.view(batch_size, input_size, 1))
        out_l = torch.bmm(weight_l, l.view(batch_size, input_size, 1))

        if base.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            if fn_neg_l is not None and bool(fn_neg_l.any().item()):
                out_l = torch.cat([out_l[:, :1, :], out_l[:, 1:, :].masked_fill(fn_neg_l.unsqueeze(-1), -1e9)], dim=1)
            if fn_neg_ab is not None and bool(fn_neg_ab.any().item()):
                out_ab = torch.cat([out_ab[:, :1, :], out_ab[:, 1:, :].masked_fill(fn_neg_ab.unsqueeze(-1), -1e9)], dim=1)
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
            out_l = torch.div(out_l, Z_l)
            out_ab = torch.div(out_ab, Z_ab)
            if fn_neg_l is not None and bool(fn_neg_l.any().item()):
                out_l = torch.cat([out_l[:, :1, :], out_l[:, 1:, :].masked_fill(fn_neg_l.unsqueeze(-1), 0.0)], dim=1)
            if fn_neg_ab is not None and bool(fn_neg_ab.any().item()):
                out_ab = torch.cat([out_ab[:, :1, :], out_ab[:, 1:, :].masked_fill(fn_neg_ab.unsqueeze(-1), 0.0)], dim=1)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()

        with torch.no_grad():
            l_pos = torch.index_select(base.memory, 0, y_flat)
            pi_mem_f64 = r_batch.pow(2).view(batch_size, 1).to(dtype=torch.float64, device=l_pos.device).clamp(0.0, 1.0)
            one_minus_m_f64 = (1.0 - momentum) * pi_mem_f64
            m_f64 = 1.0 - one_minus_m_f64
            l_pos.mul_(m_f64.to(dtype=l_pos.dtype)).add_(torch.mul(v_sym.to(dtype=l_pos.dtype), one_minus_m_f64.to(dtype=l_pos.dtype)))
            l_pos = F.normalize(l_pos, dim=1, eps=1e-12)
            base.memory.index_copy_(0, y_flat, l_pos)

        return out_l, out_ab, w_l.detach(), w_ab.detach()


def rch_v3_contrastive_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    w_batch: Optional[torch.Tensor] = None,
    margin: float = 0.2,
    shift: float = 0.1,
    tau: float = 1.0,
    false_neg_z: Optional[torch.Tensor] = None,
    false_neg_conf: Optional[torch.Tensor] = None,
    z_fn_thr: float = 1.25,
) -> torch.Tensor:
    """支持真实 Z-score 假阴性剔除 (-1e9 + logsumexp) 与固定均值归一化 ('n') 的对称跨模态排序损失。"""
    scores = im.mm(s.t())
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask_s = (scores >= (d1 - float(margin))).float().detach()
    mask_im = (scores >= (d2 - float(margin))).float().detach()
    fn_mask: Optional[torch.Tensor] = None
    if false_neg_z is not None:
        eye = torch.eye(int(scores.shape[0]), device=scores.device, dtype=torch.bool)
        conf_ok = (false_neg_conf.detach() > 0.40) if false_neg_conf is not None else torch.ones_like(eye)
        fn_mask = (~eye) & conf_ok & (false_neg_z.detach() > float(z_fn_thr))
        mask_s = mask_s.masked_fill(fn_mask, 0.0)
        mask_im = mask_im.masked_fill(fn_mask, 0.0)

    cost_s = scores * mask_s + (1.0 - mask_s) * (scores - float(shift))
    cost_im = scores * mask_im + (1.0 - mask_im) * (scores - float(shift))
    diag_s = cost_s.diag()
    diag_im = cost_im.diag()
    if fn_mask is not None and bool(fn_mask.any().item()):
        cost_s = cost_s.masked_fill(fn_mask, -1e9)
        cost_im = cost_im.masked_fill(fn_mask, -1e9)

    tau_f = float(tau)
    row_terms = -diag_s + tau_f * torch.logsumexp(cost_s / tau_f, dim=1) + float(margin)
    col_terms = -diag_im + tau_f * torch.logsumexp(cost_im / tau_f, dim=0) + float(margin)

    if w_batch is None:
        return row_terms.mean() + col_terms.mean()
    w_flat = w_batch.detach().view(-1).to(dtype=row_terms.dtype, device=row_terms.device)
    return (w_flat * row_terms).mean() + (w_flat * col_terms).mean()


def rch_v3_neighbor_nce_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    z_ii: torch.Tensor,
    z_tt: torch.Tensor,
    z_it: torch.Tensor,
    omega_ii: torch.Tensor,
    omega_tt: torch.Tensor,
    omega_it: torch.Tensor,
    top_k: int = 6,
    z_min: float = 1.35,
    fn_z: float = 0.80,
    tau_g: float = 0.50,
    tau_t: float = 1.0,
    quant_mix: float = 0.0,
) -> torch.Tensor:
    """多正例软近邻 InfoNCE 损失（过滤无效零词袋负例、防稀疏批梯度放大，可选半量化混合桥接）。"""
    b_size = int(im.shape[0])
    if b_size < 4:
        return (im * 0.0).sum() + (s * 0.0).sum()
    eye = torch.eye(b_size, device=im.device, dtype=torch.bool)
    k = min(int(top_k), max(b_size - 1, 1))
    qm = float(quant_mix)
    inv_sqrt_bit = 1.0 / math.sqrt(max(float(im.shape[1]), 1.0))

    def _pair_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cont = a.mm(b.t())
        if qm <= 0.0:
            return cont
        semi_q = 0.5 * (a.mm(b.detach().sign().t()) + a.detach().sign().mm(b.t())) * inv_sqrt_bit
        return (1.0 - qm) * cont + qm * semi_q

    def _dir_nce(pred_sim: torch.Tensor, z_mat: torch.Tensor, omega_mat: torch.Tensor) -> torch.Tensor:
        cand_mask = (~eye) & (omega_mat > 0.35) & (z_mat > float(z_min))
        neg_mask = (~eye) & (omega_mat > 0.0) & (z_mat <= float(fn_z))
        row_active_geom = cand_mask.any(dim=1) & neg_mask.any(dim=1)
        if not bool(row_active_geom.any().item()):
            return (pred_sim * 0.0).sum()

        scores_z = z_mat.masked_fill(~cand_mask, -1e9)
        top_z, top_idx = torch.topk(scores_z, k=k, dim=1)
        valid_top = top_z > -1e8

        top_om = omega_mat.gather(1, top_idx)
        q_logits = (top_z / float(tau_t)).masked_fill(~valid_top, -1e9)
        q_raw = F.softmax(q_logits, dim=1) * valid_top.to(pred_sim.dtype) * top_om.to(pred_sim.dtype)
        q_sum = q_raw.sum(dim=1, keepdim=True)
        active = row_active_geom & (q_sum.squeeze(1) > 1e-6)
        if not bool(active.any().item()):
            return (pred_sim * 0.0).sum()

        q_norm = q_raw / q_sum.clamp_min(1e-6)
        row_conf = (top_om * valid_top.to(top_om.dtype)).max(dim=1).values

        logits_all = pred_sim / float(tau_g)
        neg_logits = logits_all.masked_fill(~neg_mask, -1e9)
        log_neg_sum = torch.logsumexp(neg_logits, dim=1, keepdim=True)
        pos_logits = logits_all.gather(1, top_idx)

        per_pos_loss = F.softplus(log_neg_sum - pos_logits)
        per_row = (q_norm * per_pos_loss).sum(dim=1)
        w_row = torch.where(active, row_conf.to(pred_sim.dtype), torch.zeros_like(per_row))
        denom = w_row.sum().clamp_min(0.5 * float(b_size))
        return (w_row * per_row).sum() / denom

    li = _dir_nce(_pair_sim(im, im), z_ii, omega_ii)
    lt = _dir_nce(_pair_sim(s, s), z_tt, omega_tt)
    sim_it = _pair_sim(im, s)
    l_it = 0.5 * (_dir_nce(sim_it, z_it, omega_it) + _dir_nce(sim_it.t(), z_it.t(), omega_it.t()))
    return 0.25 * li + 0.25 * lt + 0.50 * l_it


def rch_v4_positive_graph_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    z_ii: torch.Tensor,
    z_tt: torch.Tensor,
    z_it: torch.Tensor,
    omega_ii: torch.Tensor,
    omega_tt: torch.Tensor,
    omega_it: torch.Tensor,
    z_min: float = 0.80,
    target_base: float = 0.45,
    target_slope: float = 0.20,
    target_max: float = 0.85,
    quant_mix: float = 0.25,
) -> torch.Tensor:
    """RCH-v4 单侧下界正向亲和图损失（零负排斥保护共现同类对，零上界压缩防止已对齐高相似对回拉）。"""
    b_size = int(im.shape[0])
    if b_size < 2:
        return (im * 0.0).sum() + (s * 0.0).sum()
    eye = torch.eye(b_size, device=im.device, dtype=torch.bool)
    qm = float(quant_mix)
    denom = float(b_size * b_size)

    def _pair_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cont = a.mm(b.t())
        if qm <= 0.0:
            return cont
        a_q = F.normalize(a.detach().sign(), dim=1, eps=1e-8)
        b_q = F.normalize(b.detach().sign(), dim=1, eps=1e-8)
        semi_q = 0.5 * (a.mm(b_q.t()) + a_q.mm(b.t()))
        return (1.0 - qm) * cont + qm * semi_q

    def _pos_hinge(pred_sim: torch.Tensor, z_mat: torch.Tensor, omega_mat: torch.Tensor) -> torch.Tensor:
        z_det = z_mat.detach().to(dtype=pred_sim.dtype, device=pred_sim.device)
        om_det = omega_mat.detach().to(dtype=pred_sim.dtype, device=pred_sim.device)
        pos_mask = (~eye) & (om_det > 0.35) & (z_det > float(z_min))
        if not bool(pos_mask.any().item()):
            return (pred_sim * 0.0).sum()
        z_excess = z_det - float(z_min)
        target_sim = (float(target_base) + float(target_slope) * z_excess).clamp(float(target_base), float(target_max))
        w_pos = torch.where(pos_mask, om_det * torch.sigmoid(z_excess), torch.zeros_like(z_det))
        return (w_pos * F.relu(target_sim - pred_sim).pow(2)).sum() / denom

    li = _pos_hinge(_pair_sim(im, im), z_ii, omega_ii)
    lt = _pos_hinge(_pair_sim(s, s), z_tt, omega_tt)
    sim_it = _pair_sim(im, s)
    l_it = 0.5 * (_pos_hinge(sim_it, z_it, omega_it) + _pos_hinge(sim_it.t(), z_it.t(), omega_it.t()))
    return 0.25 * li + 0.25 * lt + 0.50 * l_it


class RCHv5NCEAverage(nn.Module):
    """RCH-v5 零负例删除（保留 100% 硬负例球面均匀性）+ Epoch-0 干净门控记忆库 + 双孪生正锚路由 NCE 包装器。

    核心性质：
      1. 绝不对 ``out_l[:, 1:]`` / ``out_ab[:, 1:]`` 做 ``-1e9`` 硬负例剔除，且无 512 维 K=4096 张量采集开销（纯 O(B*K) 速度）；
      2. 干净数据（``r_batch is None`` 或 ``r_batch == 1.0``）直接零开销委托 ``base_nce`` 并返回全 1 权重；
      3. Epoch 0 预热期（``epoch < 0``）返回 ``w_l = w_ab = 1.0``，同时以 ``pi_mem = r_batch^2`` 保护记忆槽免受噪声锁定；
      4. 正式轮次（``epoch >= 0``）按 ``w_l = r_batch^2, w_ab = r_batch^2`` 初始化，独立按 ``z_ii`` / ``z_tt`` 路由单模态干净孪生
         并救援单模态损失权重至 ``0.75 * r_twin``。
    """

    def __init__(self, base_nce: Any) -> None:
        super().__init__()
        self.base = base_nce
        self.base_nce = base_nce

    @property
    def memory(self) -> torch.Tensor:
        return self.base_nce.memory

    @property
    def params(self) -> torch.Tensor:
        return self.base_nce.params

    @property
    def multinomial(self) -> Any:
        return self.base_nce.multinomial

    def forward(
        self,
        l: torch.Tensor,
        ab: torch.Tensor,
        y: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        epoch: Optional[int] = 0,
        r_batch: Optional[torch.Tensor] = None,
        valid_batch: Optional[torch.Tensor] = None,
        z_ii: Optional[torch.Tensor] = None,
        z_tt: Optional[torch.Tensor] = None,
        z_twin_thr: float = 1.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(l.size(0))
        if r_batch is None or bool(torch.all(r_batch == 1.0).item()):
            out_l_clean, out_ab_clean = self.base_nce(l, ab, y, idx=idx, epoch=epoch)
            ones = torch.ones(batch_size, device=l.device, dtype=torch.float32)
            return out_l_clean, out_ab_clean, ones, ones

        base = self.base_nce
        K = int(base.params[0].item())
        T = float(base.params[1].item())
        Z_l = float(base.params[2].item())
        Z_ab = float(base.params[3].item())
        is_warmup = (epoch is not None) and (int(epoch) < 0)
        momentum = 0.0 if is_warmup else float(base.params[4].item())

        output_size = int(base.memory.size(0))
        input_size = int(base.memory.size(1))
        y_flat = y.view(-1)

        r_b = r_batch.detach().view(batch_size).to(device=l.device, dtype=torch.float32).clamp(0.0, 1.0)
        valid_b = (
            (valid_batch.detach().view(batch_size).to(device=l.device) > 0.5)
            if valid_batch is not None
            else torch.ones(batch_size, device=l.device, dtype=torch.bool)
        )
        v_sym = F.normalize(0.5 * (l.detach() + ab.detach()), dim=1, eps=1e-12)

        if is_warmup:
            w_l = torch.ones_like(r_b).detach()
            w_ab = torch.ones_like(r_b).detach()
            w_inb = (l + ab) / 2.0
            row_arange = torch.arange(batch_size, device=l.device)
            inx_grid = torch.stack([row_arange] * batch_size)
            eye_b = torch.eye(batch_size, device=l.device, dtype=torch.bool)
            inx = torch.cat([row_arange.view(-1, 1), inx_grid[~eye_b].view(batch_size, -1)], dim=1).view(-1)
            weight = w_inb.index_select(0, inx).view(batch_size, batch_size, input_size).clone().sign_()
            weight_l = weight
            weight_ab = weight
        else:
            w_l = r_b.pow(2).clone().detach()
            w_ab = r_b.pow(2).clone().detach()
            if idx is None:
                drawn = base.multinomial.draw(batch_size * (base.K + 1))
                sample_idx: torch.Tensor = drawn.view(batch_size, -1)
                sample_idx.select(1, 0).copy_(y_flat.data)
            else:
                sample_idx = idx
            mem_raw = (
                torch.index_select(base.memory, 0, sample_idx.view(-1))
                .detach()
                .view(batch_size, K + 1, input_size)
            )
            weight_l = mem_raw.clone().sign_()
            weight_ab = mem_raw.clone().sign_()

            if z_ii is not None and batch_size >= 2:
                eye = torch.eye(batch_size, device=l.device, dtype=torch.bool)
                clean_cand_i = (r_b > 0.65).unsqueeze(0) & (~eye)
                z_ii_m = z_ii.detach().to(device=l.device, dtype=torch.float32).masked_fill(~clean_cand_i, -1e9)
                best_z_i, best_j_i = z_ii_m.max(dim=1)
                rescue_i = (r_b < 0.55) & (best_z_i > float(z_twin_thr))
                if bool(rescue_i.any().item()):
                    src_i = torch.nonzero(rescue_i, as_tuple=False).view(-1)
                    twin_j_i = best_j_i.index_select(0, src_i)
                    pos_proto_i = base.memory.index_select(0, y_flat.index_select(0, twin_j_i)).detach().to(weight_l.dtype).sign()
                    weight_l[src_i, 0, :] = pos_proto_i
                    w_l[src_i] = torch.maximum(w_l.index_select(0, src_i), 0.75 * r_b.index_select(0, twin_j_i))

            if z_tt is not None and batch_size >= 2:
                eye = torch.eye(batch_size, device=l.device, dtype=torch.bool)
                clean_cand_t = ((r_b > 0.65) & valid_b).unsqueeze(0) & (~eye)
                z_tt_m = z_tt.detach().to(device=l.device, dtype=torch.float32).masked_fill(~clean_cand_t, -1e9)
                best_z_t, best_j_t = z_tt_m.max(dim=1)
                rescue_t = valid_b & (r_b < 0.55) & (best_z_t > float(z_twin_thr))
                if bool(rescue_t.any().item()):
                    src_t = torch.nonzero(rescue_t, as_tuple=False).view(-1)
                    twin_j_t = best_j_t.index_select(0, src_t)
                    pos_proto_t = base.memory.index_select(0, y_flat.index_select(0, twin_j_t)).detach().to(weight_ab.dtype).sign()
                    weight_ab[src_t, 0, :] = pos_proto_t
                    w_ab[src_t] = torch.maximum(w_ab.index_select(0, src_t), 0.75 * r_b.index_select(0, twin_j_t))

        out_ab = torch.bmm(weight_ab, ab.view(batch_size, input_size, 1))
        out_l = torch.bmm(weight_l, l.view(batch_size, input_size, 1))

        if base.use_softmax:
            out_ab = torch.div(out_ab, T).contiguous()
            out_l = torch.div(out_l, T).contiguous()
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
            l_pos = torch.index_select(base.memory, 0, y_flat)
            pi_mem_f64 = r_b.pow(2).view(batch_size, 1).to(dtype=torch.float64, device=l_pos.device).clamp(0.0, 1.0)
            one_minus_m_f64 = (1.0 - momentum) * pi_mem_f64
            m_f64 = 1.0 - one_minus_m_f64
            l_pos.mul_(m_f64.to(dtype=l_pos.dtype)).add_(torch.mul(v_sym.to(dtype=l_pos.dtype), one_minus_m_f64.to(dtype=l_pos.dtype)))
            l_pos = F.normalize(l_pos, dim=1, eps=1e-12)
            base.memory.index_copy_(0, y_flat, l_pos)

        return out_l, out_ab, w_l.detach(), w_ab.detach()


def rch_v5_contrastive_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    w_pair: Optional[torch.Tensor] = None,
    margin: float = 0.2,
    shift: float = 0.1,
    tau: float = 1.0,
    false_neg_sim: Optional[torch.Tensor] = None,
    fn_thr: float = 0.25,
    normalize: str = "n",
) -> torch.Tensor:
    """RCH-v5/v6 成对排序损失：严格按成对可信度 w_pair 加权（支持 'n' 与 'sum_w' 归一化），温和 shift 切换保留硬负例均匀性。"""
    scores = im.mm(s.t())
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask_s = (scores >= (d1 - float(margin))).float().detach()
    mask_im = (scores >= (d2 - float(margin))).float().detach()
    if false_neg_sim is not None:
        eye = torch.eye(int(scores.shape[0]), device=scores.device, dtype=torch.bool)
        fn_mask = (~eye) & (false_neg_sim.detach() > float(fn_thr))
        mask_s = mask_s.masked_fill(fn_mask, 0.0)
        mask_im = mask_im.masked_fill(fn_mask, 0.0)

    cost_s = scores * mask_s + (1.0 - mask_s) * (scores - float(shift))
    cost_im = scores * mask_im + (1.0 - mask_im) * (scores - float(shift))

    tau_f = float(tau)
    row_terms = -cost_s.diag() + tau_f * torch.logsumexp(cost_s / tau_f, dim=1) + float(margin)
    col_terms = -cost_im.diag() + tau_f * torch.logsumexp(cost_im / tau_f, dim=0) + float(margin)

    if w_pair is None:
        return row_terms.mean() + col_terms.mean()
    w_flat = w_pair.detach().view(-1).to(dtype=row_terms.dtype, device=row_terms.device)
    if normalize == "n":
        return (w_flat * row_terms).mean() + (w_flat * col_terms).mean()
    if normalize == "sum_w":
        denom = w_flat.sum().clamp_min(1e-6)
        return (w_flat * row_terms).sum() / denom + (w_flat * col_terms).sum() / denom
    raise ValueError(f"normalize 仅支持 'n' 或 'sum_w'，当前为 {normalize}")


def compute_v5_clean_affinity_and_loss(
    im_codes: torch.Tensor,
    txt_codes: torch.Tensor,
    imgs: torch.Tensor,
    txts: torch.Tensor,
    r_batch: torch.Tensor,
    valid_batch: torch.Tensor,
    nbr_idx_i: torch.Tensor,
    nbr_w_i: torch.Tensor,
    nbr_idx_t: torch.Tensor,
    nbr_w_t: torch.Tensor,
    state: NACGAnchorState,
    stretch: float = 1.25,
    ho_weight: float = 0.30,
    upper_weight: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RCH-v5 干净锚点桥接 + 二跳亲和增强 + 非对称正向图回归损失（兼容稠密文本与稀疏词袋文本）。"""
    dev = im_codes.device
    b_size = int(im_codes.shape[0])
    if b_size < 2:
        zero_l = (im_codes * 0.0).sum() + (txt_codes * 0.0).sum()
        return zero_l, torch.zeros((b_size, b_size), device=dev, dtype=torch.float32)

    if int(state.norm_images.numel()) <= 0:
        raise ValueError("compute_v5_clean_affinity_and_loss 需要 build_nacg_anchor_state(..., store_full_norm=True)")

    eye = torch.eye(b_size, device=dev, dtype=torch.bool)
    with torch.no_grad():
        v_bool = valid_batch.detach().view(b_size).to(device=dev) > 0.5
        v_col = v_bool.unsqueeze(1).to(torch.float32)
        v_row = v_bool.unsqueeze(0).to(torch.float32)
        r_flat = r_batch.detach().view(b_size).to(device=dev, dtype=torch.float32).clamp(0.0, 1.0)
        r_col = r_flat.unsqueeze(1)
        r_col_v = r_col * v_col

        anc_img = state.norm_images.index_select(0, state.anchor_idx.to(device=state.norm_images.device)).to(
            device=dev, dtype=torch.float32
        )
        x_bridge = gather_bridged_features(
            anc_img,
            nbr_idx_t.to(device=dev),
            nbr_w_t.to(device=dev, dtype=torch.float32),
        )

        if state.is_sparse_bow and state.idf_y is not None:
            if state.norm_texts_idf is None or int(state.norm_texts_idf.numel()) <= 0:
                raise ValueError("稀疏 BoW 模式下需要 state.norm_texts_idf（store_full_norm=True）")
            anc_txt = state.norm_texts_idf.index_select(0, state.anchor_idx.to(device=state.norm_texts_idf.device)).to(
                device=dev, dtype=torch.float32
            )
            y_bridge = gather_bridged_features(
                anc_txt,
                nbr_idx_i.to(device=dev),
                nbr_w_i.to(device=dev, dtype=torch.float32),
            )
            idf_dev = state.idf_y.to(device=dev, dtype=torch.float32)
            tx_idf = (
                F.normalize(torch.log1p(txts.to(device=dev, dtype=torch.float32).clamp_min(0.0)) * idf_dev, dim=1, eps=1e-8)
                * v_col
            )
            y_for_i = F.normalize(r_col_v * tx_idf + (1.0 - r_col_v) * y_bridge, dim=1, eps=1e-8)
            s_tt = tx_idf.mm(tx_idf.t()) * 1.5 - 0.15
            c_t = y_for_i.mm(tx_idf.t()) * 1.5 - 0.15
            y_for_i_sim = y_for_i.mm(y_for_i.t()) * 1.5 - 0.15
        else:
            if int(state.norm_texts.numel()) <= 0:
                raise ValueError("稠密文本模式下需要 state.norm_texts（store_full_norm=True）")
            anc_txt = state.norm_texts.index_select(0, state.anchor_idx.to(device=state.norm_texts.device)).to(
                device=dev, dtype=torch.float32
            )
            y_bridge = gather_bridged_features(
                anc_txt,
                nbr_idx_i.to(device=dev),
                nbr_w_i.to(device=dev, dtype=torch.float32),
            )
            tx_f = F.normalize(txts.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8) * v_col
            y_for_i = F.normalize(r_col_v * tx_f + (1.0 - r_col_v) * y_bridge, dim=1, eps=1e-8)
            s_tt = tx_f.mm(tx_f.t()) * 2.0 - 1.0
            c_t = y_for_i.mm(tx_f.t()) * 2.0 - 1.0
            y_for_i_sim = y_for_i.mm(y_for_i.t()) * 2.0 - 1.0

        im_f = F.normalize(imgs.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8)
        x_for_t = F.normalize(r_col_v * im_f + (1.0 - r_col_v) * x_bridge, dim=1, eps=1e-8)
        s_ii = im_f.mm(im_f.t()) * 2.0 - 1.0
        c_i = im_f.mm(x_for_t.t()) * 2.0 - 1.0
        x_for_t_sim = x_for_t.mm(x_for_t.t()) * 2.0 - 1.0

        ho = float(ho_weight)
        r_clean = r_flat.pow(2).unsqueeze(0)
        p_i = F.normalize(F.relu(s_ii) * r_clean, dim=1, eps=1e-8)
        p_t = F.normalize(F.relu(s_tt) * (r_clean * v_row), dim=1, eps=1e-8) * v_col

        base_it = 0.5 * (c_i + c_t)
        sim_it = ((1.0 - ho) * base_it + ho * p_i.mm(p_t.t())) * v_row
        sim_ii = (1.0 - ho) * (0.5 * (s_ii + y_for_i_sim)) + ho * p_i.mm(p_i.t())
        sim_tt = ((1.0 - ho) * (0.5 * (s_tt + x_for_t_sim)) + ho * p_t.mm(p_t.t())) * (v_col * v_row)

    ones_pair = torch.ones((b_size, b_size), device=dev, dtype=torch.bool)
    valid_tt = v_bool.unsqueeze(1) & v_bool.unsqueeze(0)
    valid_it = ones_pair & v_bool.unsqueeze(0)
    st = float(stretch)
    uw = float(upper_weight)

    def _asym_pos_mse(pred: torch.Tensor, target_raw: torch.Tensor, valid_pair: torch.Tensor) -> torch.Tensor:
        pos_mask = (~eye) & valid_pair & (target_raw > 0.05)
        if not bool(pos_mask.any().item()):
            return (pred * 0.0).sum()
        target = (st * target_raw).clamp(0.0, 0.90).to(dtype=pred.dtype)
        err_sq = (target - pred).square()
        diff_sq = torch.where(pred < target, err_sq, uw * err_sq)
        masked = torch.where(pos_mask, diff_sq, torch.zeros_like(diff_sq))
        return masked.sum() / pos_mask.to(dtype=pred.dtype).sum().clamp_min(1.0)

    li = _asym_pos_mse(im_codes.mm(im_codes.t()), sim_ii, ones_pair)
    lt = _asym_pos_mse(txt_codes.mm(txt_codes.t()), sim_tt, valid_tt)
    pred_it = im_codes.mm(txt_codes.t())
    l_cross = 0.5 * (_asym_pos_mse(pred_it, sim_it, valid_it) + _asym_pos_mse(pred_it.t(), sim_it.t(), valid_it.t()))
    loss_graph = (li + lt + l_cross) / 3.0
    return loss_graph, sim_it.detach()


def compute_v6_clean_affinity_and_loss(
    im_codes: torch.Tensor,
    txt_codes: torch.Tensor,
    imgs: torch.Tensor,
    txts: torch.Tensor,
    r_batch: torch.Tensor,
    valid_batch: torch.Tensor,
    nbr_idx_i: torch.Tensor,
    nbr_w_i: torch.Tensor,
    nbr_idx_t: torch.Tensor,
    nbr_w_t: torch.Tensor,
    state: NACGAnchorState,
    ho_weight: float = 0.25,
    bridge_sharpen: float = 2.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RCH-v6 锐化干净锚点桥接 + 零均值有符号二跳亲和轮廓（消除 F.relu 假阳性正偏置）+ 模态自适应正向图回归损失。"""
    dev = im_codes.device
    b_size = int(im_codes.shape[0])
    if b_size < 2:
        zero_l = (im_codes * 0.0).sum() + (txt_codes * 0.0).sum()
        return zero_l, torch.zeros((b_size, b_size), device=dev, dtype=torch.float32)

    if int(state.norm_images.numel()) <= 0:
        raise ValueError("compute_v6_clean_affinity_and_loss 需要 build_nacg_anchor_state(..., store_full_norm=True)")

    eye = torch.eye(b_size, device=dev, dtype=torch.bool)
    is_bow = bool(state.is_sparse_bow and state.idf_y is not None)
    with torch.no_grad():
        v_bool = valid_batch.detach().view(b_size).to(device=dev) > 0.5
        v_col = v_bool.unsqueeze(1).to(torch.float32)
        v_row = v_bool.unsqueeze(0).to(torch.float32)
        r_flat = r_batch.detach().view(b_size).to(device=dev, dtype=torch.float32).clamp(0.0, 1.0)
        r_col = r_flat.unsqueeze(1)
        r_col_v = r_col * v_col

        bs_val = float(bridge_sharpen)
        if abs(bs_val - 1.0) <= 1e-8:
            w_i_raw = nbr_w_i.to(device=dev, dtype=torch.float32).clamp_min(1e-8)
            w_i_sharp = w_i_raw / w_i_raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
            w_t_raw = nbr_w_t.to(device=dev, dtype=torch.float32).clamp_min(1e-8)
            w_t_sharp = w_t_raw / w_t_raw.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            w_i_sharp = F.softmax(torch.log(nbr_w_i.to(device=dev, dtype=torch.float32).clamp_min(1e-8)) * bs_val, dim=1)
            w_t_sharp = F.softmax(torch.log(nbr_w_t.to(device=dev, dtype=torch.float32).clamp_min(1e-8)) * bs_val, dim=1)

        anc_img = state.norm_images.index_select(0, state.anchor_idx.to(device=state.norm_images.device)).to(
            device=dev, dtype=torch.float32
        )
        x_bridge = gather_bridged_features(anc_img, nbr_idx_t.to(device=dev), w_t_sharp)

        if is_bow:
            assert state.idf_y is not None
            if state.norm_texts_idf is None or int(state.norm_texts_idf.numel()) <= 0:
                raise ValueError("稀疏 BoW 模式下需要 state.norm_texts_idf（store_full_norm=True）")
            anc_txt = state.norm_texts_idf.index_select(0, state.anchor_idx.to(device=state.norm_texts_idf.device)).to(
                device=dev, dtype=torch.float32
            )
            y_bridge = gather_bridged_features(anc_txt, nbr_idx_i.to(device=dev), w_i_sharp)
            idf_dev = state.idf_y.to(device=dev, dtype=torch.float32)
            tx_idf = (
                F.normalize(torch.log1p(txts.to(device=dev, dtype=torch.float32).clamp_min(0.0)) * idf_dev, dim=1, eps=1e-8)
                * v_col
            )
            y_for_i = F.normalize(r_col_v * tx_idf + (1.0 - r_col_v) * y_bridge, dim=1, eps=1e-8)
            s_tt = tx_idf.mm(tx_idf.t()) * 1.5 - 0.15
            c_t = y_for_i.mm(tx_idf.t()) * 1.5 - 0.15
            y_for_i_sim = y_for_i.mm(y_for_i.t()) * 1.5 - 0.15
        else:
            if int(state.norm_texts.numel()) <= 0:
                raise ValueError("稠密文本模式下需要 state.norm_texts（store_full_norm=True）")
            anc_txt = state.norm_texts.index_select(0, state.anchor_idx.to(device=state.norm_texts.device)).to(
                device=dev, dtype=torch.float32
            )
            y_bridge = gather_bridged_features(anc_txt, nbr_idx_i.to(device=dev), w_i_sharp)
            tx_f = F.normalize(txts.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8) * v_col
            y_for_i = F.normalize(r_col_v * tx_f + (1.0 - r_col_v) * y_bridge, dim=1, eps=1e-8)
            s_tt = tx_f.mm(tx_f.t()) * 2.0 - 1.0
            c_t = y_for_i.mm(tx_f.t()) * 2.0 - 1.0
            y_for_i_sim = y_for_i.mm(y_for_i.t()) * 2.0 - 1.0

        im_f = F.normalize(imgs.to(device=dev, dtype=torch.float32), dim=1, eps=1e-8)
        x_for_t = F.normalize(r_col_v * im_f + (1.0 - r_col_v) * x_bridge, dim=1, eps=1e-8)
        s_ii = im_f.mm(im_f.t()) * 2.0 - 1.0
        c_i = im_f.mm(x_for_t.t()) * 2.0 - 1.0
        x_for_t_sim = x_for_t.mm(x_for_t.t()) * 2.0 - 1.0

        # 零均值有符号二跳亲和轮廓：屏蔽对角线并在有效非对角邻域内逐行零中心化，使无关对内积期望为 0
        off_eye = (~eye).to(torch.float32)
        r_clean = r_flat.pow(2).unsqueeze(0)
        off_cnt_i = off_eye.sum(dim=1, keepdim=True).clamp_min(1.0)
        mu_row_i = (s_ii * off_eye).sum(dim=1, keepdim=True) / off_cnt_i
        prof_i = F.normalize(((s_ii - mu_row_i) * off_eye) * r_clean, dim=1, eps=1e-8)

        valid_off_t = off_eye * v_row
        off_cnt_t = valid_off_t.sum(dim=1, keepdim=True).clamp_min(1.0)
        mu_row_t = (s_tt * valid_off_t).sum(dim=1, keepdim=True) / off_cnt_t
        prof_t = F.normalize(((s_tt - mu_row_t) * valid_off_t) * r_clean, dim=1, eps=1e-8) * v_col

        ho = float(ho_weight)
        base_it = 0.5 * (c_i + c_t)
        sim_it = ((1.0 - ho) * base_it + ho * prof_i.mm(prof_t.t())) * v_row
        sim_ii = (1.0 - ho) * (0.5 * (s_ii + y_for_i_sim)) + ho * prof_i.mm(prof_i.t())
        sim_tt = ((1.0 - ho) * (0.5 * (s_tt + x_for_t_sim)) + ho * prof_t.mm(prof_t.t())) * (v_col * v_row)

    ones_pair = torch.ones((b_size, b_size), device=dev, dtype=torch.bool)
    valid_tt = v_bool.unsqueeze(1) & v_bool.unsqueeze(0)
    valid_it = ones_pair & v_bool.unsqueeze(0)

    if is_bow:
        pos_thr = 0.05
        st = 1.15
        uw = 0.15
        max_t = 0.90
    else:
        pos_thr = 0.0
        st = 1.0
        uw = 1.0
        max_t = 1.0

    def _pos_reg(pred: torch.Tensor, target_raw: torch.Tensor, valid_pair: torch.Tensor) -> torch.Tensor:
        pos_mask = (~eye) & valid_pair & (target_raw > pos_thr)
        if not bool(pos_mask.any().item()):
            return (pred * 0.0).sum()
        target = (st * target_raw).clamp(0.0, max_t).to(dtype=pred.dtype)
        err_sq = (target - pred).square()
        if uw >= 1.0 - 1e-8:
            diff_sq = err_sq
        else:
            diff_sq = torch.where(pred < target, err_sq, uw * err_sq)
        masked = torch.where(pos_mask, diff_sq, torch.zeros_like(diff_sq))
        return masked.sum() / pos_mask.to(dtype=pred.dtype).sum().clamp_min(1.0)

    li = _pos_reg(im_codes.mm(im_codes.t()), sim_ii, ones_pair)
    lt = _pos_reg(txt_codes.mm(txt_codes.t()), sim_tt, valid_tt)
    pred_it = im_codes.mm(txt_codes.t())
    l_cross = 0.5 * (_pos_reg(pred_it, sim_it, valid_it) + _pos_reg(pred_it.t(), sim_it.t(), valid_it.t()))
    loss_graph = (li + lt + l_cross) / 3.0
    return loss_graph, sim_it.detach()
