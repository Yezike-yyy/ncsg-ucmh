"""PR-15 测试集二值哈希码导出、SHA-256 校验与汉明排序 PR / Top-K 曲线独立复算工具。

提供三种可追溯运行模式：
1. `--mode from_npz`：从已有 `test_codes.npz` 独立重算双向官方汉明距离排序、`MAP@ALL`、
   21 点插值/等召回率 `Precision-Recall` 曲线与 `Precision@K` 曲线，并支持校验 `.npz` 的
   `SHA-256` 及核对 `summary.json` 中的 `test_i2t_official / test_t2i_official`（误差容限 `1e-12`）。
2. `--mode generate_pr15`：在挂载特征数据后一键运行 `PR-15` 全部 15 组（3 数据集 × 5 方法，
   `64-bit, eta=0.4, seed=0`），导出并默认完整保留原始 `test_codes.npz` 二值码文件，同时生成逐组 `pr_topk.json`。
3. `--mode merge_dir`：汇总目录内 15 组运行（若存在 `test_codes.npz` 则优先从二值码重算，否则校验现有 `pr_topk.json`），
   生成与绘图脚本直接对接的合并文件 `pr_topk_curves_64bit_eta04.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore[import-not-found]
import scipy.spatial.distance  # type: ignore[import-not-found]

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from colab.full198_matrix import DATASETS, build_command, make_spec  # type: ignore[import-not-found]
from colab.run_full198 import stage_dataset  # type: ignore[import-not-found]

RECALL_GRID: List[float] = [round(0.05 * i, 2) for i in range(21)]
TOPK_GRID_DEFAULT: List[int] = [1, 10, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000]
PR15_METHODS: Tuple[str, ...] = ("off", "A-K1", "DJSRH", "JDSH", "RCH-v6")


@dataclass(frozen=True)
class PR15Spec:
    dataset: str
    data_name: str
    data_subdir: str
    variant: str
    bit: int = 64
    noise_rate: float = 0.4
    seed: int = 0

    @property
    def run_key(self) -> str:
        return f"{self.data_name}_n{self.noise_rate:g}_b{self.bit}_s{self.seed}_{self.variant}"


def build_pr15_specs() -> List[PR15Spec]:
    specs: List[PR15Spec] = []
    for ds_short, (_, data_name, subdir) in DATASETS.items():
        for variant in PR15_METHODS:
            specs.append(
                PR15Spec(
                    dataset=ds_short,
                    data_name=data_name,
                    data_subdir=subdir,
                    variant=variant,
                )
            )
    return specs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def probability(value: Any) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"指标超出 [0, 1]: {value!r}")
    return float(value)


def validate_pr15_summary(spec: PR15Spec, summary: Dict[str, Any]) -> None:
    parsed = summary["parsed_args"]
    for key in ("data_name", "variant", "bit", "seed", "noise_rate"):
        expected = getattr(spec, key)
        value = summary.get(key, parsed.get(key))
        if type(value) is not type(expected) or value != expected or parsed.get(key, expected) != expected:
            raise ValueError(f"{spec.run_key}: 摘要 {key} 不匹配")
    if summary.get("noise_seed") != 0 or parsed.get("max_epochs") != 20:
        raise ValueError(f"{spec.run_key}: 置换种子或训练轮数不匹配")
    if spec.variant == "A-K1" and any(parsed.get(k) != v for k, v in
                                        {"alpha": 0.7, "shift": 0.1, "num_workers": 2}.items()):
        raise ValueError(f"{spec.run_key}: A-K1 配置不匹配")
    if spec.variant == "RCH-v6" and parsed.get("nacg_force_one") is not False:
        raise ValueError(f"{spec.run_key}: RCH-v6 可靠度开关不匹配")
    i2t, t2i, avg = (probability(summary[k]) for k in
                     ("test_i2t_official", "test_t2i_official", "test_avg_official"))
    if abs(avg - (i2t + t2i) / 2) > 1e-12:
        raise ValueError(f"{spec.run_key}: 摘要双向均值不一致")


def validate_pr15_curve(spec: PR15Spec, curve: Dict[str, Any], summary: Dict[str, Any]) -> None:
    for key in ("data_name", "variant", "bit", "seed", "noise_rate"):
        expected = getattr(spec, key)
        if type(curve.get(key)) is not type(expected) or curve[key] != expected:
            raise ValueError(f"{spec.run_key}: 曲线 {key} 不匹配")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(curve.get("test_codes_sha256", ""))):
        raise ValueError(f"{spec.run_key}: 缺少有效的原始码指纹")
    if type(curve.get("test_codes_bytes")) is not int or curve["test_codes_bytes"] <= 0:
        raise ValueError(f"{spec.run_key}: 原始码文件大小无效")
    for key in ("test_i2t_official", "test_t2i_official", "test_avg_official"):
        if abs(probability(curve[key]) - summary[key]) > 1e-12:
            raise ValueError(f"{spec.run_key}: 曲线与摘要 {key} 不一致")
    for key in ("i2t", "t2i"):
        direction = curve[key]
        n_db, n_queries = direction["n_db"], direction["n_valid_queries"]
        if (type(n_db) is not int or n_db <= 0 or n_db != summary["n_db"] or
                type(n_queries) is not int or not 0 < n_queries <= summary["n_query"]):
            raise ValueError(f"{spec.run_key}/{key}: 检索库或查询数不匹配")
        delta = abs(probability(direction["map_official_recomputed"]) - curve[f"test_{key}_official"])
        recorded_delta = probability(curve["map_recompute_abs_diff"][key])
        if delta > 1e-12 or abs(delta - recorded_delta) > 1e-12:
            raise ValueError(f"{spec.run_key}/{key}: MAP 复算差异不一致")
        topk = [k for k in TOPK_GRID_DEFAULT if k <= n_db]
        if direction["recall_grid"] != RECALL_GRID or direction["topk_grid"] != topk:
            raise ValueError(f"{spec.run_key}/{key}: 曲线网格不匹配")
        for field, grid in (("precision_at_recall", RECALL_GRID), ("precision_interpolated", RECALL_GRID),
                            ("precision_at_k", topk)):
            values = direction[field]
            if not isinstance(values, list) or len(values) != len(grid):
                raise ValueError(f"{spec.run_key}/{key}: {field} 长度不匹配")
            for value in values:
                probability(value)
        interp = direction["precision_interpolated"]
        if any(left < right for left, right in zip(interp, interp[1:])):
            raise ValueError(f"{spec.run_key}/{key}: 插值精度曲线不单调")


def read_pr15_run(run_dir: Path, spec: PR15Spec) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_pr15_summary(spec, summary)
    curve_path = run_dir / "pr_topk.json"
    archived = None
    if curve_path.is_file():
        archived = json.loads(curve_path.read_text(encoding="utf-8"))
        validate_pr15_curve(spec, archived, summary)
    codes = run_dir / "test_codes.npz"
    if not codes.is_file():
        if archived is None:
            raise FileNotFoundError(f"{spec.run_key}: 缺少曲线及原始码")
        return archived
    recomputed = recompute_from_npz(codes, summary_path=summary_path,
                                    expected_sha256=archived["test_codes_sha256"] if archived else None)
    validate_pr15_curve(spec, recomputed, summary)
    if archived:
        if archived["test_codes_bytes"] != recomputed["test_codes_bytes"]:
            raise ValueError(f"{spec.run_key}: 原始码文件大小与归档不一致")
        for key in ("i2t", "t2i"):
            for field in ("n_valid_queries", "n_db", "precision_at_recall", "precision_interpolated", "precision_at_k"):
                if archived[key][field] != recomputed[key][field]:
                    raise ValueError(f"{spec.run_key}/{key}: 原始码复算 {field} 与归档不一致")
    return recomputed


def load_codes_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """兼容 `nacg_ucch_train.py` / `i1_train.py` 与 `run_external.py` 两种 `.npz` 键名约定。"""
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"retrieval_imgs", "retrieval_txts", "retrieval_labs", "query_imgs", "query_txts", "query_labs"} <= keys:
            return {
                "retrieval_images": np.asarray(data["retrieval_imgs"]),
                "retrieval_texts": np.asarray(data["retrieval_txts"]),
                "retrieval_labels": np.asarray(data["retrieval_labs"]),
                "query_images": np.asarray(data["query_imgs"]),
                "query_texts": np.asarray(data["query_txts"]),
                "query_labels": np.asarray(data["query_labs"]),
            }
        if {
            "retrieval_images",
            "retrieval_texts",
            "retrieval_labels",
            "query_images",
            "query_texts",
            "query_labels",
        } <= keys:
            return {
                "retrieval_images": np.asarray(data["retrieval_images"]),
                "retrieval_texts": np.asarray(data["retrieval_texts"]),
                "retrieval_labels": np.asarray(data["retrieval_labels"]),
                "query_images": np.asarray(data["query_images"]),
                "query_texts": np.asarray(data["query_texts"]),
                "query_labels": np.asarray(data["query_labels"]),
            }
    raise KeyError(f"无法识别 {npz_path} 中的二值码键集合")


def compute_direction_pr_topk(
    retrieval_codes: np.ndarray,
    query_codes: np.ndarray,
    retrieval_labels: np.ndarray,
    query_labels: np.ndarray,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """按官方汉明距离排序（`distances.argsort()`）精确计算单方向 MAP@ALL、21 点 PR 与 Top-K 曲线。"""
    n_db = int(retrieval_labels.shape[0])
    n_query = int(query_labels.shape[0])
    if n_db <= 0 or n_query <= 0:
        raise ValueError("检索库或查询集样本数必须为正")

    topk_grid = [k for k in TOPK_GRID_DEFAULT if k <= n_db]
    recall_arr = np.asarray(RECALL_GRID, dtype=np.float64)
    ranks = np.arange(1.0, 1.0 + n_db, dtype=np.float64)

    ap_list: List[float] = []
    interp_rows: List[np.ndarray] = []
    at_recall_rows: List[np.ndarray] = []
    topk_rows: List[np.ndarray] = []

    r_lab_f = np.asarray(retrieval_labels, dtype=np.float64)
    q_lab_f = np.asarray(query_labels, dtype=np.float64)

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        distances = scipy.spatial.distance.cdist(
            query_codes[start:end],
            retrieval_codes,
            metric="hamming",
        )
        official_order = distances.argsort()
        del distances

        rel_block = (q_lab_f[start:end] @ r_lab_f.T) > 0
        for row_i in range(end - start):
            rel_ordered = rel_block[row_i, official_order[row_i]]
            total_pos = int(rel_ordered.sum())
            if total_pos == 0:
                continue
            prec_all = rel_ordered.cumsum(dtype=np.float64) / ranks
            ap_list.append(float(np.dot(rel_ordered, prec_all) / float(total_pos)))

            rel_pos = np.flatnonzero(rel_ordered)
            rec_at_rel = np.arange(1, total_pos + 1, dtype=np.float64) / float(total_pos)
            prec_at_rel = prec_all[rel_pos]
            interp_at_rel = np.maximum.accumulate(prec_at_rel[::-1])[::-1]

            idx = np.searchsorted(rec_at_rel, recall_arr, side="left")
            idx = np.clip(idx, 0, total_pos - 1)
            interp_rows.append(interp_at_rel[idx])
            at_recall_rows.append(prec_at_rel[idx])
            topk_rows.append(np.asarray([prec_all[k - 1] for k in topk_grid], dtype=np.float64))

    if not ap_list:
        raise ValueError("没有包含至少一个相关检索项的有效查询")

    return {
        "n_valid_queries": len(ap_list),
        "n_db": n_db,
        "map_official_recomputed": float(np.mean(ap_list)),
        "recall_grid": RECALL_GRID,
        "precision_interpolated": [round(float(x), 6) for x in np.mean(interp_rows, axis=0)],
        "precision_at_recall": [round(float(x), 6) for x in np.mean(at_recall_rows, axis=0)],
        "topk_grid": topk_grid,
        "precision_at_k": [round(float(x), 6) for x in np.mean(topk_rows, axis=0)],
    }


def recompute_from_npz(
    npz_path: Path,
    *,
    summary_path: Optional[Path] = None,
    expected_sha256: Optional[str] = None,
    data_name: Optional[str] = None,
    variant: Optional[str] = None,
    bit: Optional[int] = None,
    noise_rate: Optional[float] = None,
    seed: Optional[int] = None,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    if not npz_path.is_file() or npz_path.stat().st_size <= 0:
        raise FileNotFoundError(f"二值码文件不存在或为空: {npz_path}")

    actual_sha = sha256_file(npz_path)
    if expected_sha256 is not None and actual_sha.lower() != expected_sha256.strip().lower():
        raise ValueError(f"test_codes.npz SHA-256 不匹配: actual={actual_sha}, expected={expected_sha256}")

    summary_meta: Dict[str, Any] = {}
    if summary_path is not None:
        summary_meta = json.loads(summary_path.read_text(encoding="utf-8"))

    arrays = load_codes_npz(npz_path)
    i2t_res = compute_direction_pr_topk(
        arrays["retrieval_texts"],
        arrays["query_images"],
        arrays["retrieval_labels"],
        arrays["query_labels"],
        chunk_size=chunk_size,
    )
    t2i_res = compute_direction_pr_topk(
        arrays["retrieval_images"],
        arrays["query_texts"],
        arrays["retrieval_labels"],
        arrays["query_labels"],
        chunk_size=chunk_size,
    )

    i2t_map = float(i2t_res["map_official_recomputed"])
    t2i_map = float(t2i_res["map_official_recomputed"])
    ref_i2t = probability(summary_meta["test_i2t_official"]) if summary_path is not None else i2t_map
    ref_t2i = probability(summary_meta["test_t2i_official"]) if summary_path is not None else t2i_map
    diff_i2t = abs(i2t_map - ref_i2t)
    diff_t2i = abs(t2i_map - ref_t2i)
    if summary_meta and max(diff_i2t, diff_t2i) > 1e-12:
        raise ValueError(
            f"从 test_codes.npz 重算的 MAP 与 summary.json 不一致: diff_i2t={diff_i2t}, diff_t2i={diff_t2i}"
        )

    parsed = summary_meta.get("parsed_args") if isinstance(summary_meta.get("parsed_args"), dict) else {}
    resolved_bit = bit if bit is not None else int(summary_meta.get("bit", parsed.get("bit", arrays["query_images"].shape[1])))
    resolved_dn = data_name or str(summary_meta.get("data_name", parsed.get("data_name", "unknown")))
    resolved_var = variant or str(summary_meta.get("variant", "unknown"))
    resolved_nr = float(noise_rate if noise_rate is not None else summary_meta.get("noise_rate", 0.4))
    resolved_seed = int(seed if seed is not None else summary_meta.get("seed", 0))

    return {
        "data_name": resolved_dn,
        "variant": resolved_var,
        "bit": int(resolved_bit),
        "noise_rate": resolved_nr,
        "seed": resolved_seed,
        "test_i2t_official": ref_i2t,
        "test_t2i_official": ref_t2i,
        "test_avg_official": 0.5 * (ref_i2t + ref_t2i),
        "test_codes_sha256": actual_sha,
        "test_codes_bytes": int(npz_path.stat().st_size),
        "map_recompute_abs_diff": {
            "i2t": float(diff_i2t),
            "t2i": float(diff_t2i),
        },
        "i2t": i2t_res,
        "t2i": t2i_res,
    }


def build_pr15_train_command(
    spec: PR15Spec,
    baseline_dir: Path,
    local_data_root: Path,
    out_dir: Path,
    python_bin: str = "python3",
) -> List[str]:
    if spec.variant in ("off", "RCH-v6"):
        cmd = [
            python_bin,
            str(baseline_dir / "ours/nacg_ucch_train.py"),
            "--data_name", spec.data_name,
            "--data_root", str(local_data_root),
            "--out", str(out_dir),
            "--variant", spec.variant,
            "--seed", str(spec.seed),
            "--noise_rate", str(spec.noise_rate),
            "--noise_seed", "0",
            "--bit", str(spec.bit),
            "--num_workers", "0",
            "--no_save_ckpt",
            "--dump_codes",
        ]
        if spec.variant == "RCH-v6":
            cmd.extend(["--alpha", "0.65", "--lambda_graph", "0.35", "--rch_v6_bridge_sharpen", "2.5"])
        return cmd

    if spec.variant in ("A-K1", "DJSRH", "JDSH"):
        baseline_spec = make_spec(spec.dataset, spec.variant, spec.bit, spec.noise_rate, spec.seed)
        cmd = build_command(baseline_spec, baseline_dir, local_data_root, out_dir, python_bin, num_workers=2)
        cmd.append("--dump_codes")
        if spec.variant == "A-K1":
            cmd.append("test_codes.npz")
        else:
            # PR 导出工具沿用外部训练器的设备自动选择。
            cmd.remove("--require_cuda")
        return cmd

    raise ValueError(f"未知的 PR-15 方法变体: {spec.variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("from_npz", "generate_pr15", "merge_dir", "verify_historical_npz"),
        required=True,
    )
    parser.add_argument("--npz", type=Path, default=None, help="from_npz 模式输入 test_codes.npz 路径")
    parser.add_argument("--npz_root", type=Path, default=None, help="verify_historical_npz 模式查找原始 test_codes.npz 的根目录")
    parser.add_argument("--summary", type=Path, default=None, help="可选 summary.json 路径，用于交叉核对 MAP")
    parser.add_argument("--expected_sha256", type=str, default=None, help="可选 test_codes.npz 预期 SHA-256")
    parser.add_argument("--runs_root", type=Path, default=None, help="merge_dir / verify_historical_npz 历史归档运行根目录")
    parser.add_argument("--out", type=Path, default=None, help="输出 JSON 文件路径")
    parser.add_argument("--baseline_dir", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--data_root", type=Path, default=Path("/content/drive/MyDrive/UCCH/data"))
    parser.add_argument("--local_data_root", type=Path, default=Path("/content/data"))
    parser.add_argument("--out_root", type=Path, default=Path("/content/drive/MyDrive/NACG_pr15_runs"))
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--keep_npz", dest="keep_npz", action="store_true", default=True)
    parser.add_argument("--no_keep_npz", dest="keep_npz", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "from_npz":
        if args.npz is None or args.out is None:
            raise ValueError("--mode from_npz 需要同时提供 --npz 与 --out")
        payload = recompute_from_npz(
            args.npz,
            summary_path=args.summary,
            expected_sha256=args.expected_sha256,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "OK", "out": str(args.out), "sha256": payload["test_codes_sha256"]}))
        return

    if args.mode == "verify_historical_npz":
        if args.runs_root is None or args.out is None:
            raise ValueError("--mode verify_historical_npz 需要同时提供 --runs_root 与 --out")
        search_root = args.npz_root if args.npz_root is not None else args.runs_root
        per_run: Dict[str, Any] = {}
        missing_keys: List[str] = []
        failed_keys: List[str] = []
        verified_count = 0
        for spec in build_pr15_specs():
            arch_dir = args.runs_root / spec.run_key
            sum_p = arch_dir / "summary.json"
            pr_p = arch_dir / "pr_topk.json"
            summary_obj = json.loads(sum_p.read_text(encoding="utf-8"))
            validate_pr15_summary(spec, summary_obj)
            arch_curve = json.loads(pr_p.read_text(encoding="utf-8"))
            validate_pr15_curve(spec, arch_curve, summary_obj)
            cand_npz = search_root / spec.run_key / "test_codes.npz"
            if not cand_npz.is_file():
                missing_keys.append(spec.run_key)
                per_run[spec.run_key] = {
                    "status": "MISSING_NPZ",
                    "expected_sha256": arch_curve["test_codes_sha256"],
                    "expected_bytes": arch_curve["test_codes_bytes"],
                }
                continue
            try:
                recomputed = recompute_from_npz(
                    cand_npz,
                    summary_path=sum_p,
                    expected_sha256=arch_curve["test_codes_sha256"],
                    data_name=spec.data_name,
                    variant=spec.variant,
                    bit=spec.bit,
                    noise_rate=spec.noise_rate,
                    seed=spec.seed,
                )
                validate_pr15_curve(spec, recomputed, summary_obj)
                if arch_curve["test_codes_bytes"] != recomputed["test_codes_bytes"]:
                    raise ValueError("test_codes_bytes 不一致")
                for d_key in ("i2t", "t2i"):
                    for fld in ("n_valid_queries", "n_db", "precision_at_recall", "precision_interpolated", "precision_at_k"):
                        if arch_curve[d_key][fld] != recomputed[d_key][fld]:
                            raise ValueError(f"{d_key}/{fld} 与历史曲线不一致")
                verified_count += 1
                per_run[spec.run_key] = {
                    "status": "VERIFIED",
                    "sha256": recomputed["test_codes_sha256"],
                    "bytes": recomputed["test_codes_bytes"],
                }
            except Exception as exc:
                failed_keys.append(spec.run_key)
                per_run[spec.run_key] = {"status": "FAILED", "error": str(exc)}

        all_ok = (verified_count == 15 and not missing_keys and not failed_keys)
        report = {
            "status": "VERIFIED_ALL_15" if all_ok else ("FAILED_MISMATCH" if failed_keys else "MISSING_HISTORICAL_NPZ"),
            "verified_count": verified_count,
            "missing_count": len(missing_keys),
            "failed_count": len(failed_keys),
            "missing_keys": missing_keys,
            "failed_keys": failed_keys,
            "per_run": per_run,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "per_run"}, ensure_ascii=False))
        if not all_ok:
            sys.exit(1 if failed_keys else 2)
        return

    if args.mode == "merge_dir":
        if args.runs_root is None or args.out is None:
            raise ValueError("--mode merge_dir 需要同时提供 --runs_root 与 --out")
        merged: Dict[str, Any] = {}
        for spec in build_pr15_specs():
            r_dir = args.runs_root / spec.run_key
            merged[spec.run_key] = read_pr15_run(r_dir, spec)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "OK", "merged_runs": len(merged), "out": str(args.out)}))
        return

    specs = build_pr15_specs()
    if args.dry_run:
        for spec in specs:
            cmd = build_pr15_train_command(
                spec,
                args.baseline_dir,
                args.local_data_root,
                args.out_root / spec.run_key,
                args.python_bin,
            )
            print(shlex.join(cmd))
        print(json.dumps({"dry_run": True, "runs": len(specs), "keep_npz": bool(args.keep_npz)}))
        return

    for spec in {s.dataset: s for s in specs}.values():
        stage_dataset(args.data_root / spec.data_subdir, args.local_data_root / spec.data_subdir)
    args.out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for spec in specs:
        run_dir = args.out_root / spec.run_key
        run_dir.mkdir(parents=True, exist_ok=True)
        npz_p = run_dir / "test_codes.npz"
        sum_p = run_dir / "summary.json"
        if not (npz_p.is_file() and sum_p.is_file()):
            if sum_p.exists() or npz_p.exists() or (run_dir / "pr_topk.json").exists():
                raise ValueError(f"{spec.run_key}: 已有历史或不完整产物，请在独立输出目录重新导出")
            cmd = build_pr15_train_command(spec, args.baseline_dir, args.local_data_root, run_dir, args.python_bin)
            with (run_dir / "train.log").open("w", encoding="utf-8") as log_f:
                res = subprocess.run(cmd, cwd=str(args.baseline_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=False)
            if res.returncode != 0:
                raise RuntimeError(f"{spec.run_key} 运行失败 (exit={res.returncode})")
        pr_payload = read_pr15_run(run_dir, spec)
        (run_dir / "pr_topk.json").write_text(json.dumps(pr_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.keep_npz and npz_p.is_file():
            npz_p.unlink()
        print(f"COMPLETED {spec.run_key} keep_npz={args.keep_npz}", flush=True)


if __name__ == "__main__":
    main()
