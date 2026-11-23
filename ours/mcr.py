"""MCR（Mask-Consistency Reliability）核心算法模块。

包含两个互补模块与跨模态噪声对应注入工具：
  - 模块 A（掩码一致性可信度估计与可靠加权）：
      1. 跨模态掩码视图一致性信号 c_i = 0.5 * (<h^x_i, h̃^y_i> + <h̃^x_i, h^y_i>)
         或逐样本排序损失信号 L_r,i；
      2. 二分量高斯混合模型（GMM）无监督拟合干净/噪声分布，输出干净分量后验 π_i；
      3. 样本权重 w_i = π_i * (1 - <h^x_i, h^y_i>)^γ + (1 - π_i) * δ（全程 detach）；
      4. 加权对比哈希损失 L_c（NCESoftmaxLoss 逐样本加权）与加权跨模态排序损失 L_r（ContrastiveLoss 逐样本行列加权）；
      5. 可靠动量记忆库更新 m_i = 1 - (1 - m) * π_i（包装/继承上游 NCEAverage，不改上游源码）。
  - 模块 B（模态主导度估计与自适应不对称掩码对比学习）：
      1. 模态主导度 d_i = Δx_i / (Δx_i + Δy_i + ε)，其中 Δx_i = max(0, <h^x, h^y> - <h̃^x, h^y>)，
         Δy_i = max(0, <h^x, h^y> - <h^x, h̃^y>)，当 Δx_i + Δy_i < ε 时 d_i = 0.5，支持 EMA 平滑；
      2. 逐样本不对称掩码保留率 ρ^x_i = clip(ρ̄ - κ(d_i - 0.5), ρ_min, ρ_max)，
         ρ^y_i = clip(ρ̄ + κ(d_i - 0.5), ρ_min, ρ_max)；
      3. 掩码视图对称 InfoNCE 损失 L_m(h̃^x, h̃^y)，支持模块 A 的样本权重 w_i 加权。
  - 跨模态对应噪声注入：
      使用独立 RandomState(noise_seed) 选取 round(η * n) 个训练样本，对其文本特征做无不动点随机置换（错位），
      保证检索库与测试查询集完全不受污染。
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def make_derangement(rng: np.random.RandomState, m: int) -> np.ndarray:
    """使用独立 RandomState 生成长度为 m (m >= 2) 的无不动点随机置换（Derangement）。

    先生成随机置换 perm，若存在不动点 perm[i] == i，则将其与循环相邻位置 (i + 1) % m 交换。
    由于 perm 是双射且 m >= 2，交换前 perm[(i + 1) % m] != i，交换后位置 i 与 (i + 1) % m 均不可能是自身，
    单趟扫描即可严格保证任意 i 均满足 perm[i] != i。
    """
    if m < 2:
        raise ValueError(f"无不动点置换要求样本数 m >= 2，当前 m={m}")
    perm = rng.permutation(m).astype(np.int64)
    for i in range(m):
        if perm[i] == i:
            nxt = (i + 1) % m
            perm[i], perm[nxt] = perm[nxt], perm[i]
    return perm


def build_correspondence_permutation(
    n_data: int,
    noise_rate: float,
    noise_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回文本置换掩码及来源索引；未置换位置的来源索引为 -1。"""
    noise_mask = np.zeros(n_data, dtype=bool)
    source_indices = np.full(n_data, -1, dtype=np.int64)
    if noise_rate <= 0.0:
        return noise_mask, source_indices
    if noise_rate > 1.0:
        raise ValueError(f"noise_rate 必须在 [0, 1] 范围内，当前为 {noise_rate}")

    n_noisy = int(round(float(noise_rate) * n_data))
    if n_noisy <= 0:
        return noise_mask, source_indices
    if n_noisy == 1:
        raise ValueError(
            f"选中的噪声样本数 round({noise_rate} * {n_data}) = 1，无法在单样本内构造无不动点置换"
        )

    rng = np.random.RandomState(int(noise_seed))
    noisy_indices = rng.choice(n_data, size=n_noisy, replace=False)
    perm = make_derangement(rng, n_noisy)
    source_indices[noisy_indices] = noisy_indices[perm]
    noise_mask[noisy_indices] = True
    return noise_mask, source_indices


def inject_correspondence_noise(
    train_dataset: Any,
    noise_rate: float,
    noise_seed: int = 0,
) -> np.ndarray:
    """按独立种子置换训练文本，返回表示置换位置的布尔掩码。"""
    noise_mask, source_indices = build_correspondence_permutation(
        len(train_dataset),
        noise_rate,
        noise_seed,
    )
    if not np.any(noise_mask):
        return noise_mask

    orig_texts = np.array(train_dataset.texts, copy=True)
    new_texts = np.array(train_dataset.texts, copy=True)
    new_texts[noise_mask] = orig_texts[source_indices[noise_mask]]
    train_dataset.texts = new_texts
    return noise_mask


def apply_feature_mask(
    x: torch.Tensor,
    keep_prob: Union[float, torch.Tensor],
    generator: torch.Generator,
    rescale: bool = False,
) -> torch.Tensor:
    """对输入特征张量 x 施加逐维独立的 Bernoulli(keep_prob) 二值掩码。

    参数：
      x: 形状为 (B, d) 的输入特征张量。
      keep_prob: 标量保留率 ρ，或形状为 (B,) / (B, 1) 的逐样本保留率张量。
      generator: 独立的 torch.Generator，避免污染全局随机数流。
      rescale: 是否按 1 / keep_prob 做期望缩放（默认 False，即 x̃ = x ⊙ m）。
    """
    gen_device = generator.device
    if isinstance(keep_prob, (int, float)):
        rho_val = float(max(0.0, min(1.0, float(keep_prob))))
        probs = torch.full(x.shape, rho_val, dtype=x.dtype, device=gen_device)
    else:
        kp = keep_prob.detach().to(device=gen_device, dtype=x.dtype).view(-1, 1).clamp(min=0.0, max=1.0)
        probs = kp.expand(x.shape)

    mask = torch.bernoulli(probs, generator=generator).to(device=x.device, dtype=x.dtype)
    masked = x * mask

    if rescale:
        if isinstance(keep_prob, (int, float)):
            denom = max(float(keep_prob), 1e-6)
            masked = masked / denom
        else:
            denom_t = keep_prob.detach().to(device=x.device, dtype=x.dtype).view(-1, 1).clamp(min=1e-6)
            masked = masked / denom_t

    return masked


def compute_consistency_signal(
    h_x: torch.Tensor,
    h_y: torch.Tensor,
    h_x_tilde: torch.Tensor,
    h_y_tilde: torch.Tensor,
) -> torch.Tensor:
    """计算逐样本跨模态掩码视图一致性信号 c_i = 0.5 * (<h^x_i, h̃^y_i> + <h̃^x_i, h^y_i>)。"""
    sim_x_ytilde = (h_x * h_y_tilde).sum(dim=1)
    sim_xtilde_y = (h_x_tilde * h_y).sum(dim=1)
    return 0.5 * (sim_x_ytilde + sim_xtilde_y)


def compute_per_sample_lr_signal(
    im: torch.Tensor,
    s: torch.Tensor,
    margin: float = 0.2,
    shift: float = 0.1,
    tau: float = 1.0,
) -> torch.Tensor:
    """计算上游 ContrastiveLoss (L_r) 的逐样本损失项（行项 + 列项，值越小代表对应越干净）。"""
    scores = im.mm(s.t())
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask_s = (scores >= (d1 - margin)).float().detach()
    cost_s = scores * mask_s + (1.0 - mask_s) * (scores - shift)
    mask_im = (scores >= (d2 - margin)).float().detach()
    cost_im = scores * mask_im + (1.0 - mask_im) * (scores - shift)

    row_terms = -cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + margin
    col_terms = -cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + margin
    return (row_terms + col_terms).detach()


def fit_gmm_reliability(
    signals: np.ndarray,
    mode: Literal["consistency", "loss"] = "consistency",
    random_state: int = 1123,
) -> Tuple[np.ndarray, float, float]:
    """对一维信号做 min-max 归一化并拟合 2 分量 GMM，返回干净分量后验概率 π 及(干净均值, 噪声均值)。

    GMM 参数固定为：n_components=2, max_iter=20, reg_covar=5e-4, tol=1e-2, random_state=固定值。
    """
    sig = np.asarray(signals, dtype=np.float64).reshape(-1)
    s_min = float(np.min(sig))
    s_max = float(np.max(sig))
    if s_max - s_min < 1e-12:
        ones = np.ones(sig.shape[0], dtype=np.float32)
        return ones, 0.0, 0.0

    s_norm = ((sig - s_min) / (s_max - s_min)).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        max_iter=20,
        reg_covar=5e-4,
        tol=1e-2,
        random_state=int(random_state),
    )
    gmm.fit(s_norm)
    means = np.asarray(gmm.means_, dtype=np.float64).reshape(-1)

    if mode == "consistency":
        clean_comp = int(np.argmax(means))
    elif mode == "loss":
        clean_comp = int(np.argmin(means))
    else:
        raise ValueError(f"未知的信号模式 mode={mode}，仅支持 'consistency' 或 'loss'")

    noisy_comp = 1 - clean_comp
    posteriors = gmm.predict_proba(s_norm)
    pi = posteriors[:, clean_comp].astype(np.float32)
    return pi, float(means[clean_comp]), float(means[noisy_comp])


def compute_sample_weights(
    pi: torch.Tensor,
    h_x: torch.Tensor,
    h_y: torch.Tensor,
    gamma: float = 0.0,
    delta: float = 0.1,
) -> torch.Tensor:
    """计算样本权重 w_i = π_i * (1 - <h^x_i, h^y_i>)^γ + (1 - π_i) * δ，全程 detach。"""
    pi_det = pi.detach().view(-1).to(dtype=h_x.dtype, device=h_x.device)
    sim = (h_x.detach() * h_y.detach()).sum(dim=1).view(-1)
    one_minus_sim = (1.0 - sim).clamp(min=0.0, max=2.0)

    if gamma == 0.0:
        focal = torch.ones_like(one_minus_sim)
    else:
        focal = one_minus_sim.pow(float(gamma))

    w = pi_det * focal + (1.0 - pi_det) * float(delta)
    return w.detach()


def weighted_nce_softmax_loss(
    x: torch.Tensor,
    w: torch.Tensor,
    normalize: Literal["n", "sum_w"] = "n",
) -> torch.Tensor:
    """按样本权重 w 加权的 NCESoftmaxLoss。

    当 w ≡ 1 且 normalize='n' 时，计算式 (-x.softmax(1)[:, 0].log() * 1).mean() 与上游 NCESoftmaxLoss 完全一致。
    """
    probs = x.softmax(1)
    per_sample = -probs[:, 0].log().view(-1)
    w_flat = w.detach().view(-1).to(dtype=per_sample.dtype, device=per_sample.device)
    weighted = w_flat * per_sample

    if normalize == "n":
        return weighted.mean()
    if normalize == "sum_w":
        return weighted.sum() / w_flat.sum().clamp(min=1e-12)
    raise ValueError(f"normalize 仅支持 'n' 或 'sum_w'，当前为 {normalize}")


def weighted_contrastive_loss(
    im: torch.Tensor,
    s: torch.Tensor,
    w: torch.Tensor,
    margin: float = 0.2,
    shift: float = 0.1,
    tau: float = 1.0,
    normalize: Literal["n", "sum_w"] = "n",
) -> torch.Tensor:
    """按样本权重 w 加权的跨模态排序损失 ContrastiveLoss (L_r)。

    复刻上游 ContrastiveLoss.forward 的数学：按行项（cost_s 第 i 行）和列项（cost_im 第 i 列）
    分别乘以 w_i 后取均值（或按 Σw 归一化）。当 w ≡ 1 且 normalize='n' 时与上游逐位一致。
    """
    scores = im.mm(s.t())
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask_s = (scores >= (d1 - margin)).float().detach()
    cost_s = scores * mask_s + (1.0 - mask_s) * (scores - shift)
    mask_im = (scores >= (d2 - margin)).float().detach()
    cost_im = scores * mask_im + (1.0 - mask_im) * (scores - shift)

    row_terms = -cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + margin
    col_terms = -cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + margin

    w_flat = w.detach().view(-1).to(dtype=row_terms.dtype, device=row_terms.device)
    if normalize == "n":
        return (w_flat * row_terms).mean() + (w_flat * col_terms).mean()
    if normalize == "sum_w":
        denom = w_flat.sum().clamp(min=1e-12)
        return (w_flat * row_terms).sum() / denom + (w_flat * col_terms).sum() / denom
    raise ValueError(f"normalize 仅支持 'n' 或 'sum_w'，当前为 {normalize}")


class MCRNCEAverage(nn.Module):
    """支持可靠度加权动量更新 m_i = 1 - (1 - m) * π_i 的记忆库包装器。

    直接复用上游 NCEAverage 实例的 memory buffer、params buffer 与 multinomial 采样器，
    仅在记忆库动量更新步骤按样本可信度 π_i 调整动量系数：
      m_i = 1 - (1 - m) * π_i
    当 π_i ≡ 1 时退化为标量动量 m；预热期 m = 0 时 m_i = 1 - π_i（噪声样本 π_i -> 0 时保持旧记忆不写入噪声）。
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
        pi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pi is None:
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

        # 按样本可信度 π_i 动态调节动量更新：m_i = 1 - (1 - m) * π_i，先在 float64 下计算以消除 π≡1 时的 1-ULP 舍入差
        with torch.no_grad():
            l_merged = (l + ab) / 2.0
            l_merged.div_(l_merged.norm(dim=1, keepdim=True))
            l_pos = torch.index_select(base.memory, 0, y.view(-1))
            pi_col_f64 = pi.detach().view(batch_size, 1).to(dtype=torch.float64, device=l_pos.device).clamp(0.0, 1.0)
            one_minus_m_f64 = (1.0 - momentum) * pi_col_f64
            m_f64 = 1.0 - one_minus_m_f64
            l_pos.mul_(m_f64.to(dtype=l_pos.dtype))
            l_pos.add_(torch.mul(l_merged, one_minus_m_f64.to(dtype=l_pos.dtype)))
            l_pos = l_pos.div_(l_pos.norm(dim=1, keepdim=True))
            base.memory.index_copy_(0, y, l_pos)

        return out_l, out_ab


def compute_modality_dominance(
    h_x: torch.Tensor,
    h_y: torch.Tensor,
    h_x_tilde: torch.Tensor,
    h_y_tilde: torch.Tensor,
    prev_d: Optional[torch.Tensor] = None,
    d_ema: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算逐样本模态主导度 d_i = Δx_i / (Δx_i + Δy_i + ε)。

    其中：
      Δx_i = clamp(<h^x_i, h^y_i> - <h̃^x_i, h^y_i>, min=0)
      Δy_i = clamp(<h^x_i, h^y_i> - <h^x_i, h̃^y_i>, min=0)
    当 Δx_i + Δy_i < ε 时，令 d_i = 0.5（对称）。
    若 prev_d 非空且 d_ema > 0，则按 d_new = d_ema * prev_d + (1 - d_ema) * d_raw 做指数移动平均。
    """
    sim_xy = (h_x * h_y).sum(dim=1)
    sim_xtilde_y = (h_x_tilde * h_y).sum(dim=1)
    sim_x_ytilde = (h_x * h_y_tilde).sum(dim=1)

    delta_x = (sim_xy - sim_xtilde_y).clamp(min=0.0)
    delta_y = (sim_xy - sim_x_ytilde).clamp(min=0.0)
    denom = delta_x + delta_y

    half = torch.full_like(denom, 0.5)
    d_raw = torch.where(denom < eps, half, delta_x / (denom + eps))

    if prev_d is not None and d_ema > 0.0:
        ema = float(d_ema)
        prev = prev_d.detach().to(device=d_raw.device, dtype=d_raw.dtype)
        d_out = ema * prev + (1.0 - ema) * d_raw
    else:
        d_out = d_raw

    return d_out.detach()


def compute_keep_probs(
    d: torch.Tensor,
    rho_bar: float = 0.8,
    kappa: float = 0.2,
    rho_min: float = 0.5,
    rho_max: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据模态主导度 d_i 计算图像与文本的不对称掩码保留概率 (ρ^x_i, ρ^y_i)。

    公式：
      ρ^x_i = clip(ρ̄ - κ * (d_i - 0.5), ρ_min, ρ_max)
      ρ^y_i = clip(ρ̄ + κ * (d_i - 0.5), ρ_min, ρ_max)
    主导模态（例如图像主导 d_i > 0.5）获得更低的保留率（被掩码更多），迫使弱势模态充分学习。
    """
    d_det = d.detach()
    offset = float(kappa) * (d_det - 0.5)
    rho_x = (float(rho_bar) - offset).clamp(min=float(rho_min), max=float(rho_max))
    rho_y = (float(rho_bar) + offset).clamp(min=float(rho_min), max=float(rho_max))
    return rho_x.detach(), rho_y.detach()


def masked_infonce_loss(
    h_x_tilde: torch.Tensor,
    h_y_tilde: torch.Tensor,
    tau_m: float = 0.2,
    w: Optional[torch.Tensor] = None,
    normalize: Literal["n", "sum_w"] = "n",
) -> torch.Tensor:
    """计算掩码视图对称 InfoNCE 损失 L_m(h̃^x, h̃^y)，可选按样本权重 w 加权。"""
    logits = torch.mm(h_x_tilde, h_y_tilde.t()) / float(tau_m)
    loss_x2y = -F.log_softmax(logits, dim=1).diag()
    loss_y2x = -F.log_softmax(logits, dim=0).diag()
    per_sample = 0.5 * (loss_x2y + loss_y2x)

    if w is None:
        return per_sample.mean()

    w_flat = w.detach().view(-1).to(dtype=per_sample.dtype, device=per_sample.device)
    weighted = w_flat * per_sample
    if normalize == "n":
        return weighted.mean()
    if normalize == "sum_w":
        return weighted.sum() / w_flat.sum().clamp(min=1e-12)
    raise ValueError(f"normalize 仅支持 'n' 或 'sum_w'，当前为 {normalize}")
