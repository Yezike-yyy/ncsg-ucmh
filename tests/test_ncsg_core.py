"""Standalone unit tests for NCSG core modules (ODNC gate, signed two-hop graph, plug-in losses).

Runs directly without external datasets or third-party upstream repositories.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.methods import method_defaults
from ours.mcr import build_correspondence_permutation
from ours.nacg import (
    build_nacg_anchor_state,
    compute_rch_sample_weights,
    compute_v6_clean_affinity_and_loss,
    djsrh_loss_nacg,
    jdsh_loss_nacg,
)


def _make_synthetic_corpus(
    n: int = 512,
    n_classes: int = 32,
    dx: int = 48,
    dy: int = 36,
    noise_rate: float = 0.0,
    seed: int = 0,
):
    """Create dense multi-cluster cross-modal features with shared semantic structure."""
    rng = np.random.default_rng(seed)
    cls_x = rng.normal(size=(n_classes, dx)).astype(np.float32) * 2.5
    cls_y = rng.normal(size=(n_classes, dy)).astype(np.float32) * 2.5
    labels = rng.integers(0, n_classes, size=n)
    x = cls_x[labels] + 0.35 * rng.normal(size=(n, dx)).astype(np.float32)
    y = cls_y[labels] + 0.35 * rng.normal(size=(n, dy)).astype(np.float32)
    # Shift dense features slightly away from exact zeros so sparse-BoW branch is not triggered
    y = y + np.where(y == 0.0, 1e-3, 0.0).astype(np.float32)
    mask, sources = build_correspondence_permutation(n, noise_rate, noise_seed=0)
    y_noisy = y.copy()
    if mask.any():
        y_noisy[mask] = y[sources[mask]]
    return torch.from_numpy(x), torch.from_numpy(y_noisy), mask


class NCSGCoreTests(unittest.TestCase):
    def test_odnc_exact_zero_noise_gate_on_clean_data(self) -> None:
        x, y, mask = _make_synthetic_corpus(n=512, noise_rate=0.0, seed=1)
        self.assertEqual(int(mask.sum()), 0)
        state = build_nacg_anchor_state(
            images_gpu=x,
            texts_gpu=y,
            seed=0,
            num_anchors=64,
            top_k_bridge=8,
            tau_null=0.12,
            ridge_reg=5.0,
        )
        self.assertEqual(float(state.hat_eta_0), 0.0)
        self.assertTrue(np.allclose(state.r.cpu().numpy(), 1.0))

    def test_odnc_detects_and_separates_noisy_correspondence(self) -> None:
        x, y, mask = _make_synthetic_corpus(n=512, noise_rate=0.4, seed=2)
        state = build_nacg_anchor_state(
            images_gpu=x,
            texts_gpu=y,
            seed=0,
            num_anchors=64,
            top_k_bridge=8,
            tau_null=0.12,
            ridge_reg=5.0,
        )
        r = state.r.cpu().numpy()
        self.assertGreater(float(state.hat_eta_0), 0.15)
        self.assertGreater(float(r[mask == 0].mean()), float(r[mask == 1].mean()) + 0.50)

    def test_signed_two_hop_graph_loss_finite_and_differentiable(self) -> None:
        torch.manual_seed(0)
        x, y, _ = _make_synthetic_corpus(n=300, dx=48, dy=36, noise_rate=0.4, seed=3)
        state = build_nacg_anchor_state(
            images_gpu=x,
            texts_gpu=y,
            seed=0,
            num_anchors=64,
            top_k_bridge=8,
            tau_null=0.12,
            ridge_reg=5.0,
        )
        b, bit = 32, 16
        idx = torch.arange(b)
        hx = F.normalize(torch.randn(b, bit, requires_grad=True), dim=1)
        hy = F.normalize(torch.randn(b, bit, requires_grad=True), dim=1)
        hx.retain_grad()
        hy.retain_grad()

        loss_g, sim_it = compute_v6_clean_affinity_and_loss(
            im_codes=hx,
            txt_codes=hy,
            imgs=x[idx],
            txts=y[idx],
            r_batch=state.r[idx],
            valid_batch=state.valid_mask[idx],
            nbr_idx_i=state.nbr_idx_I[idx],
            nbr_w_i=state.nbr_w_I[idx],
            nbr_idx_t=state.nbr_idx_T[idx],
            nbr_w_t=state.nbr_w_T[idx],
            state=state,
            ho_weight=0.25,
            bridge_sharpen=2.5,
        )
        self.assertEqual(tuple(sim_it.shape), (b, b))
        self.assertTrue(torch.isfinite(loss_g).item())
        loss_g.backward()
        self.assertIsNotNone(hx.grad)
        self.assertIsNotNone(hy.grad)
        self.assertTrue(torch.isfinite(hx.grad).all().item())
        self.assertTrue(torch.isfinite(hy.grad).all().item())

    def test_sample_weights_and_plugin_losses_run_cleanly(self) -> None:
        torch.manual_seed(42)
        r = torch.tensor([0.0, 0.5, 1.0])
        w = compute_rch_sample_weights(r, gamma_r=1.0, delta=0.08)
        self.assertTrue(torch.allclose(w, torch.tensor([0.08, 0.54, 1.0]), atol=1e-6))

        b, dx, dy, bit = 24, 32, 24, 16
        f_i = F.normalize(torch.randn(b, dx), dim=1)
        f_t = F.normalize(torch.randn(b, dy), dim=1)
        x_br = F.normalize(torch.randn(b, dx), dim=1)
        y_br = F.normalize(torch.randn(b, dy), dim=1)
        rb = torch.rand(b)
        code_i = torch.tanh(torch.randn(b, bit, requires_grad=True))
        code_t = torch.tanh(torch.randn(b, bit, requires_grad=True))

        l_djsrh, _ = djsrh_loss_nacg(
            f_i, f_t, code_i, code_t, method_defaults("DJSRH", "mscoco_fea"), rb, x_br, y_br
        )
        l_jdsh, _ = jdsh_loss_nacg(
            f_i, f_t, code_i, code_t, method_defaults("JDSH", "mscoco_fea"), rb, x_br, y_br
        )
        self.assertTrue(torch.isfinite(l_djsrh).item())
        self.assertTrue(torch.isfinite(l_jdsh).item())


if __name__ == "__main__":
    unittest.main()
