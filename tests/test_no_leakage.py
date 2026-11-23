"""防数据泄漏与零测试集超参作弊自动化审计测试。"""
from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from typing import Set

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ours.mcr import build_correspondence_permutation  # type: ignore[import-not-found]
from ours.nacg import (  # type: ignore[import-not-found]
    build_nacg_anchor_state,
    compute_centered_clean_graph,
    compute_zscore_clean_graph,
    update_nacg_epoch_reliability,
)


def _collect_ast_tokens(node: ast.AST) -> Set[str]:
    tokens: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            tokens.add(sub.id.lower())
        elif isinstance(sub, ast.Attribute):
            tokens.add(sub.attr.lower())
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            tokens.add(sub.value.lower())
    return tokens


class NoLeakageAuditTests(unittest.TestCase):
    def test_static_no_dataset_cheating_or_query_leak(self) -> None:
        # 1. ours/nacg.py 静态零泄漏审计：零标签、零查询集、零数据集名称分支
        nacg_path = ROOT / "ours" / "nacg.py"
        nacg_src = nacg_path.read_text(encoding="utf-8")
        nacg_lower = nacg_src.lower()
        for forbidden in (
            "data_name",
            "mirflickr",
            "nus_wide",
            "mscoco",
            "query_images",
            "query_texts",
            "query_labels",
            "val_avg",
            "test_avg",
            "labels",
        ):
            self.assertNotIn(
                forbidden,
                nacg_lower,
                f"Forbidden identifier {forbidden!r} found in ours/nacg.py",
            )
        nacg_tree = ast.parse(nacg_src, filename=str(nacg_path))
        for node in ast.walk(nacg_tree):
            if isinstance(node, ast.Name):
                self.assertNotIn("label", node.id.lower(), f"Label identifier {node.id!r} in ours/nacg.py:{node.lineno}")
                self.assertNotIn("query", node.id.lower(), f"Query identifier {node.id!r} in ours/nacg.py:{node.lineno}")
            elif isinstance(node, ast.Attribute):
                self.assertNotIn("label", node.attr.lower(), f"Label attr {node.attr!r} in ours/nacg.py:{node.lineno}")
                self.assertNotIn("query", node.attr.lower(), f"Query attr {node.attr!r} in ours/nacg.py:{node.lineno}")

        # 2. ours/nacg_ucch_train.py 静态 AST 审计
        train_path = ROOT / "ours" / "nacg_ucch_train.py"
        train_src = train_path.read_text(encoding="utf-8")
        tree = ast.parse(train_src, filename=str(train_path))

        # 验证 query_images / query_texts / query_labels 仅在训练循环与最佳权重恢复后访问
        load_best_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_state_dict"
        ]
        self.assertGreaterEqual(len(load_best_lines), 2, "Expected best model restore before test eval")
        test_eval_start = max(load_best_lines)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("query_images", "query_texts", "query_labels"):
                self.assertGreater(
                    node.lineno,
                    test_eval_start,
                    f"Data.{node.attr} accessed before best checkpoint restore at line {node.lineno}",
                )

        # 验证零数据集名条件分支（if / 三元表达式 / 比较表达式均不得按数据集名称或 data_name 分支）
        ds_branch_tokens = ("mirflickr", "nus_wide", "mscoco", "data_name")
        for node in ast.walk(tree):
            test_expr = None
            if isinstance(node, (ast.If, ast.IfExp)):
                test_expr = node.test
            elif isinstance(node, ast.Compare):
                test_expr = node
            if test_expr is not None:
                toks = _collect_ast_tokens(test_expr)
                for tok in toks:
                    for ds_tok in ds_branch_tokens:
                        self.assertNotIn(
                            ds_tok,
                            tok,
                            f"Dataset-specific branching on {tok!r} forbidden in nacg_ucch_train.py:{node.lineno}",
                        )

        # 验证训练期间（恢复最佳权重之前）data.labels 仅切片用于 val_labels 或 IndexedPairDataset，从不传入 nacg 或损失函数
        lines = train_src.splitlines()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "labels"
                and isinstance(node.value, ast.Name)
                and node.value.id == "data"
                and node.lineno <= test_eval_start
            ):
                ctx_window = "\n".join(lines[max(0, node.lineno - 3) : min(len(lines), node.lineno + 1)])
                self.assertTrue(
                    ("val_labels" in ctx_window) or ("IndexedPairDataset" in ctx_window),
                    f"Unexpected data.labels reference at line {node.lineno}: {lines[node.lineno - 1]}",
                )

        # 验证噪声掩码仅用于只读 AUC 诊断与产物存档，从不进入损失计算
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("mask", "sources"):
                if isinstance(node.value, ast.Name) and node.value.id == "noise":
                    ctx_window = "\n".join(lines[max(0, node.lineno - 3) : min(len(lines), node.lineno + 1)])
                    allowed = any(
                        tok in ctx_window
                        for tok in ("noise_mask=noise.mask", "np.any(noise.mask)", "n_noisy", "label_inconsistency")
                    )
                    self.assertTrue(allowed, f"Unexpected noise.{node.attr} usage at line {node.lineno}: {lines[node.lineno - 1]}")

        # 3. baselines/run_external.py 静态 AST 审计
        ext_path = ROOT / "baselines" / "run_external.py"
        ext_src = ext_path.read_text(encoding="utf-8")
        ext_tree = ast.parse(ext_src, filename=str(ext_path))
        ext_lines = ext_src.splitlines()

        restore_lines = [
            node.lineno
            for node in ast.walk(ext_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "restore_state_dict"
        ]
        self.assertGreaterEqual(len(restore_lines), 2, "Expected restore_state_dict for image & text before test eval")
        ext_test_eval_start = max(restore_lines)

        for node in ast.walk(ext_tree):
            if isinstance(node, ast.Attribute) and node.attr in ("query_images", "query_texts", "query_labels"):
                self.assertGreater(
                    node.lineno,
                    ext_test_eval_start,
                    f"External runner accessed data.{node.attr} before restore_state_dict at line {node.lineno}",
                )

        for node in ast.walk(ext_tree):
            test_expr = None
            if isinstance(node, (ast.If, ast.IfExp)):
                test_expr = node.test
            elif isinstance(node, ast.Compare):
                test_expr = node
            if test_expr is not None:
                toks = _collect_ast_tokens(test_expr)
                for tok in toks:
                    for ds_tok in ("mirflickr", "nus_wide", "mscoco"):
                        self.assertNotIn(
                            ds_tok,
                            tok,
                            f"Dataset literal branching on {tok!r} forbidden in baselines/run_external.py:{node.lineno}",
                        )

        for node in ast.walk(ext_tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "labels"
                and isinstance(node.value, ast.Name)
                and node.value.id == "data"
                and node.lineno <= ext_test_eval_start
            ):
                ext_window = "\n".join(ext_lines[max(0, node.lineno - 3) : min(len(ext_lines), node.lineno + 1)])
                self.assertTrue(
                    any(k in ext_window for k in ("FeatureDataset", "val_size", "val_labels", "validation_metrics")),
                    f"Unexpected data.labels usage before test eval in run_external.py:{node.lineno}: {ext_lines[node.lineno - 1]}",
                )

        for node in ast.walk(ext_tree):
            if isinstance(node, ast.Attribute) and node.attr in ("mask", "sources"):
                if isinstance(node.value, ast.Name) and node.value.id == "noise":
                    ext_window = "\n".join(ext_lines[max(0, node.lineno - 3) : min(len(ext_lines), node.lineno + 1)])
                    allowed_ext = any(
                        tok in ext_window
                        for tok in ("noise_mask=noise.mask", "reliability_traversal", "n_noisy")
                    )
                    self.assertTrue(allowed_ext, f"Unexpected noise.{node.attr} usage in run_external.py:{node.lineno}: {ext_lines[node.lineno - 1]}")

        # 4. colab 矩阵执行器零测试集调参与零数据集特判审计
        for runner_rel in (
            "colab/run_nacg_final51.py",
            "colab/run_nacg_sfactor18.py",
        ):
            runner_path = ROOT / runner_rel
            runner_src = runner_path.read_text(encoding="utf-8")
            runner_lower = runner_src.lower()
            for forbidden in (
                "mirflickr",
                "nus_wide",
                "mscoco",
                "query_images",
                "query_texts",
                "query_labels",
                "val_avg",
                "spec.dataset ==",
                "spec.data_name ==",
                ".labels",
            ):
                self.assertNotIn(
                    forbidden,
                    runner_lower,
                    f"Forbidden identifier {forbidden!r} found in {runner_rel}",
                )
            runner_tree = ast.parse(runner_src, filename=str(runner_path))
            for fn_node in runner_tree.body:
                if isinstance(fn_node, ast.FunctionDef) and fn_node.name != "is_completed_valid":
                    fn_tokens = _collect_ast_tokens(fn_node)
                    self.assertNotIn(
                        "test_avg_official",
                        fn_tokens,
                        f"test_avg_official only permitted inside is_completed_valid in {runner_rel}",
                    )

        # 5. colab/summary_validator.py 与 colab/build_pr_curves.py 零数据集特判审计
        for helper_rel in ("colab/summary_validator.py", "colab/build_pr_curves.py"):
            helper_path = ROOT / helper_rel
            helper_src = helper_path.read_text(encoding="utf-8")
            helper_lower = helper_src.lower()
            for ds_lit in ("mirflickr", "nus_wide", "mscoco", "spec.dataset ==", "spec.data_name =="):
                self.assertNotIn(ds_lit, helper_lower, f"Forbidden dataset literal {ds_lit!r} in {helper_rel}")
            helper_tree = ast.parse(helper_src, filename=str(helper_path))
            for node in ast.walk(helper_tree):
                if isinstance(node, (ast.If, ast.IfExp)):
                    toks = _collect_ast_tokens(node.test)
                    for ds_tok in ("mirflickr", "nus_wide", "mscoco"):
                        for tok in toks:
                            self.assertNotIn(ds_tok, tok, f"Dataset branch on {tok!r} in {helper_rel}:{node.lineno}")

    def test_noise_mask_does_not_affect_anchor_state(self) -> None:
        rng = np.random.RandomState(17)
        n_train = 400
        topics_i = rng.normal(size=(8, 64)).astype(np.float32)
        topics_t = rng.normal(size=(8, 32)).astype(np.float32)
        pick = rng.randint(0, 8, size=n_train)
        imgs = topics_i[pick] + 0.15 * rng.normal(size=(n_train, 64)).astype(np.float32)
        txts = topics_t[pick] + 0.15 * rng.normal(size=(n_train, 32)).astype(np.float32)

        mask, sources = build_correspondence_permutation(n_train, 0.4, noise_seed=0)
        txts_noisy = txts.copy()
        txts_noisy[mask] = txts[sources[mask]]

        imgs_t = torch.from_numpy(imgs)
        txts_t = torch.from_numpy(txts_noisy)

        for sc_storey in (False, True):
            st_with_mask = build_nacg_anchor_state(
                imgs_t,
                txts_t,
                seed=0,
                num_anchors=96,
                noise_mask=mask,
                self_consistent_storey=sc_storey,
            )
            st_no_mask = build_nacg_anchor_state(
                imgs_t,
                txts_t,
                seed=0,
                num_anchors=96,
                noise_mask=None,
                self_consistent_storey=sc_storey,
            )

            self.assertEqual(st_with_mask.hat_eta_0, st_no_mask.hat_eta_0)
            self.assertEqual(st_with_mask.hat_eta_raw_0, st_no_mask.hat_eta_raw_0)
            self.assertIsNotNone(st_with_mask.cold_start_auc)
            self.assertIsNone(st_no_mask.cold_start_auc)
            for attr in (
                "r_0",
                "r",
                "s_0",
                "s_0_null",
                "z_0",
                "z_0_null",
                "nbr_idx_I",
                "nbr_w_I",
                "nbr_idx_T",
                "nbr_w_T",
                "mu_x",
                "mu_y",
                "px_norm",
                "py_norm",
            ):
                self.assertTrue(
                    torch.equal(getattr(st_with_mask, attr), getattr(st_no_mask, attr)),
                    f"Attribute {attr} differed when passing noise_mask (self_consistent_storey={sc_storey})!",
                )

            g_with = compute_centered_clean_graph(
                imgs_t[:64],
                txts_t[:64],
                st_with_mask.r[:64],
                st_with_mask.valid_mask[:64],
                st_with_mask,
            )
            g_no = compute_centered_clean_graph(
                imgs_t[:64],
                txts_t[:64],
                st_no_mask.r[:64],
                st_no_mask.valid_mask[:64],
                st_no_mask,
            )
            for idx_t, (t1, t2) in enumerate(zip(g_with, g_no)):
                self.assertTrue(torch.equal(t1, t2), f"Centered graph tensor[{idx_t}] differed when passing noise_mask!")

            z_with = compute_zscore_clean_graph(
                imgs_t[:64],
                txts_t[:64],
                st_with_mask.r[:64],
                st_with_mask.valid_mask[:64],
                st_with_mask,
            )
            z_no = compute_zscore_clean_graph(
                imgs_t[:64],
                txts_t[:64],
                st_no_mask.r[:64],
                st_no_mask.valid_mask[:64],
                st_no_mask,
            )
            for idx_t, (t1, t2) in enumerate(zip(z_with, z_no)):
                self.assertTrue(torch.equal(t1, t2), f"Z-score graph tensor[{idx_t}] differed when passing noise_mask!")

    def test_noise_mask_does_not_affect_epoch_reliability_all_modes(self) -> None:
        rng = np.random.RandomState(29)
        n_train = 360
        topics_i = rng.normal(size=(8, 48)).astype(np.float32)
        topics_t = rng.normal(size=(8, 28)).astype(np.float32)
        pick = rng.randint(0, 8, size=n_train)
        imgs = topics_i[pick] + 0.12 * rng.normal(size=(n_train, 48)).astype(np.float32)
        txts_clean = topics_t[pick] + 0.12 * rng.normal(size=(n_train, 28)).astype(np.float32)

        mask, sources = build_correspondence_permutation(n_train, 0.4, noise_seed=0)
        txts_noisy = txts_clean.copy()
        txts_noisy[mask] = txts_clean[sources[mask]]

        imgs_t = torch.from_numpy(imgs)
        txts_noisy_t = torch.from_numpy(txts_noisy)
        txts_clean_t = torch.from_numpy(txts_clean)

        code_topics = rng.normal(size=(8, 16)).astype(np.float32)
        hx_base = torch.from_numpy(code_topics[pick] + 0.20 * rng.normal(size=(n_train, 16)).astype(np.float32))
        hy_clean_base = code_topics[pick] + 0.20 * rng.normal(size=(n_train, 16)).astype(np.float32)
        hy_noisy_np = hy_clean_base.copy()
        hy_noisy_np[mask] = hy_clean_base[sources[mask]]
        hy_base = torch.from_numpy(hy_noisy_np)

        mode_configs = (
            (False, False, False, False),  # standard mode
            (True, False, False, False),   # online_dominant=True
            (False, True, False, False),   # self_consistent_storey=True
            (True, True, False, False),    # online_dominant=True + self_consistent_storey=True (RCH-v4)
            (False, False, True, False),   # v5_mode=True (RCH-v5 / RCH-v5-sym)
            (False, False, True, True),    # v5_mode=True + refresh_bridges=True (RCH-v6 / RCH-v6-con)
        )
        for online_dom, sc_storey, v5_flag, ref_br in mode_configs:
            st_with = build_nacg_anchor_state(
                imgs_t,
                txts_noisy_t,
                seed=0,
                num_anchors=96,
                noise_mask=mask,
                store_full_norm=v5_flag,
                self_consistent_storey=sc_storey,
            )
            st_no = build_nacg_anchor_state(
                imgs_t,
                txts_noisy_t,
                seed=0,
                num_anchors=96,
                noise_mask=None,
                store_full_norm=v5_flag,
                self_consistent_storey=sc_storey,
            )
            init_nbr = {
                nbr_attr: getattr(st_with, nbr_attr).clone()
                for nbr_attr in ("nbr_idx_I", "nbr_w_I", "nbr_idx_T", "nbr_w_T")
            }
            for ep in (0, 1, 5):
                r_with, diag_with = update_nacg_epoch_reliability(
                    st_with,
                    hx_base,
                    hy_base,
                    epoch=ep,
                    noise_mask=mask,
                    online_dominant=online_dom,
                    self_consistent_storey=sc_storey,
                    v5_mode=v5_flag,
                    refresh_bridges=ref_br,
                )
                r_no, diag_no = update_nacg_epoch_reliability(
                    st_no,
                    hx_base,
                    hy_base,
                    epoch=ep,
                    noise_mask=None,
                    online_dominant=online_dom,
                    self_consistent_storey=sc_storey,
                    v5_mode=v5_flag,
                    refresh_bridges=ref_br,
                )
                max_diff = float((r_with - r_no).abs().max().item())
                self.assertEqual(
                    max_diff,
                    0.0,
                    f"r differed at ep={ep}, online={online_dom}, storey={sc_storey}, v5={v5_flag}, ref_br={ref_br}",
                )
                self.assertTrue(torch.equal(r_with, r_no))
                self.assertTrue(torch.equal(st_with.r, st_no.r))
                for nbr_attr in ("nbr_idx_I", "nbr_w_I", "nbr_idx_T", "nbr_w_T"):
                    self.assertTrue(
                        torch.equal(getattr(st_with, nbr_attr), getattr(st_no, nbr_attr)),
                        f"{nbr_attr} differed at ep={ep}, online={online_dom}, storey={sc_storey}, ref_br={ref_br}",
                    )
                    if not ref_br:
                        self.assertTrue(
                            torch.equal(getattr(st_with, nbr_attr), init_nbr[nbr_attr]),
                            f"{nbr_attr} mutated despite refresh_bridges=False at ep={ep}, v5={v5_flag}",
                        )
                self.assertEqual(diag_with["hat_eta"], diag_no["hat_eta"])
                self.assertEqual(diag_with["lam"], diag_no["lam"])
                self.assertEqual(diag_with.get("refresh_bridges"), diag_no.get("refresh_bridges"))
                self.assertIsNotNone(diag_with["pi_auc"])
                self.assertIsNone(diag_no["pi_auc"])

        # Clean data (hat_eta_0 == 0.0) exact gate invariance
        st_clean_with = build_nacg_anchor_state(
            imgs_t,
            txts_clean_t,
            seed=0,
            num_anchors=96,
            noise_mask=mask,
            store_full_norm=False,
            self_consistent_storey=True,
        )
        st_clean_no = build_nacg_anchor_state(
            imgs_t,
            txts_clean_t,
            seed=0,
            num_anchors=96,
            noise_mask=None,
            store_full_norm=False,
            self_consistent_storey=True,
        )
        r_cw, _ = update_nacg_epoch_reliability(
            st_clean_with,
            hx_base,
            torch.from_numpy(hy_clean_base),
            epoch=2,
            noise_mask=mask,
            online_dominant=True,
            self_consistent_storey=True,
        )
        r_cn, _ = update_nacg_epoch_reliability(
            st_clean_no,
            hx_base,
            torch.from_numpy(hy_clean_base),
            epoch=2,
            noise_mask=None,
            online_dominant=True,
            self_consistent_storey=True,
        )
        self.assertTrue(torch.equal(r_cw, r_cn))
        self.assertTrue(bool(torch.all(r_cw == 1.0).item()))


if __name__ == "__main__":
    unittest.main()
