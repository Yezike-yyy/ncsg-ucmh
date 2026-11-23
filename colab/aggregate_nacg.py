"""联立 FULL198、S3 与 NACG（Pilot / Iter-2..6 / Final-51）摘要，计算预注册 Gate、全 6 格基准超越榜、多比特表、噪声鲁棒曲线与逐步消融表。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL198 = REPO_ROOT / ".sci-skills" / "experiments" / "M5-FULL198-REMAIN192-001" / "full198_aggregate.json"
DEFAULT_S3_AGG = REPO_ROOT / ".sci-skills" / "experiments" / "M5-SUPP-S3-001" / "s3_aggregate.json"

VARIANT_ORDER: Tuple[str, ...] = (
    "off",
    "A-K1",
    "DJSRH",
    "DJSRH-pi",
    "DJSRH-nacg",
    "JDSH",
    "JDSH-pi",
    "JDSH-nacg",
    "NC-MRCH",
    "UCCH-NACG",
    "RCH-v2-base",
    "RCH-v2",
    "RCH-v3-sym",
    "RCH-v3",
    "RCH-v4-nog",
    "RCH-v4-ablate",
    "RCH-v4",
    "RCH-v5-nog",
    "RCH-v5-sym",
    "RCH-v5",
    "RCH-v6-nog",
    "RCH-v6-fnonly",
    "RCH-v6-gonly",
    "RCH-v6-stat",
    "RCH-v6-con",
    "RCH-v6",
)

BASE_MAINLINE_VARIANTS: Tuple[str, ...] = (
    "NC-MRCH",
    "UCCH-NACG",
    "RCH-v2-base",
    "RCH-v2",
    "RCH-v3-sym",
    "RCH-v3",
    "RCH-v4-nog",
    "RCH-v4-ablate",
    "RCH-v4",
    "RCH-v5-nog",
    "RCH-v5-sym",
    "RCH-v5",
    "RCH-v6-nog",
    "RCH-v6-fnonly",
    "RCH-v6-gonly",
    "RCH-v6-stat",
    "RCH-v6-con",
    "RCH-v6",
)


def _variant_sort_key(variant: str) -> Tuple[int, str]:
    try:
        return (VARIANT_ORDER.index(variant), variant)
    except ValueError:
        return (len(VARIANT_ORDER), variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nacg_root", required=True)
    parser.add_argument("--extra_roots", nargs="*", default=[])
    parser.add_argument(
        "--full198",
        default=str(DEFAULT_FULL198),
    )
    parser.add_argument(
        "--s3_agg",
        default=str(DEFAULT_S3_AGG),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records: Dict[Tuple[str, float, int, str, int], float] = {}
    anchor_records: Dict[str, float] = {}
    pi_diag_summary: Dict[Tuple[str, float, int, str], List[Dict[str, Any]]] = defaultdict(list)

    def add_record(data: Dict[str, Any], from_pilot: bool = False, source_dir_name: str = "") -> None:
        parsed = data.get("parsed_args", {})
        name = data.get("dataset", data.get("data_name", parsed.get("data_name")))
        if not name:
            return
        variant = str(data.get("variant"))
        if "test_i2t_official" in data:
            score = (float(data["test_i2t_official"]) + float(data["test_t2i_official"])) * 50.0
        elif "i2t" in data:
            score = (float(data["i2t"]) + float(data["t2i"])) / 2.0
        else:
            score = (float(data["i2t_official"]) + float(data["t2i_official"])) * 50.0

        run_key = str(data.get("run_key", ""))
        if from_pilot and (run_key.endswith("_off-nacg_anchor") or source_dir_name.endswith("_off-nacg_anchor")):
            anchor_records[str(name)] = score
            return

        bit_int = int(data.get("bit", parsed.get("bit")))
        nr_float = float(data["noise_rate"])
        seed_int = int(data["seed"])
        key = (str(name), nr_float, bit_int, variant, seed_int)
        if key in records:
            raise ValueError(f"重复运行身份: {key}")
        records[key] = score

        if from_pilot:
            nacg_d = data.get("nacg_diagnostics") or {}
            last_pi = nacg_d.get("last_epoch_pi") or {}
            pi_diag_summary[(str(name), nr_float, bit_int, variant)].append(
                {
                    "seed": seed_int,
                    "bit": bit_int,
                    "hat_eta_0": nacg_d.get("hat_eta_0"),
                    "hat_eta_raw_0": nacg_d.get("hat_eta_raw_0"),
                    "cold_start_auc": nacg_d.get("cold_start_auc"),
                    "zero_noise_exact_gate": nacg_d.get("zero_noise_exact_gate"),
                    "refresh_bridges": nacg_d.get("refresh_bridges"),
                    "ep19_auc": last_pi.get("pi_auc"),
                    "ep19_pi_mean": last_pi.get("pi_mean"),
                }
            )

    full_path = Path(args.full198)
    with full_path.open(encoding="utf-8") as handle:
        full_payload = json.load(handle)
    for rec in full_payload["per_seed_records"]:
        add_record(rec, from_pilot=False)

    s3_path = Path(args.s3_agg)
    s3_rows_map: Dict[Tuple[str, float, str], float] = {}
    if s3_path.is_file():
        with s3_path.open(encoding="utf-8") as handle:
            s3_payload = json.load(handle)
        for row in s3_payload.get("rows", []):
            s3_rows_map[(str(row["dataset"]), float(row["noise_rate"]), str(row["variant"]))] = float(row["mean_pct"])

    seen_paths: Set[Path] = set()
    for root_str in [args.nacg_root, *args.extra_roots]:
        for path in sorted(Path(root_str).rglob("summary.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            with resolved.open(encoding="utf-8") as handle:
                add_record(json.load(handle), from_pilot=True, source_dir_name=resolved.parent.name)

    discovered_v4_extra = sorted(
        {
            variant
            for (_ds, _nr, _bit, variant, _seed) in records
            if variant.startswith(("RCH-v4", "RCH-v5", "RCH-v6")) and variant not in BASE_MAINLINE_VARIANTS
        }
    )
    mainline_variants: Tuple[str, ...] = BASE_MAINLINE_VARIANTS + tuple(discovered_v4_extra)

    groups: Dict[Tuple[str, float, int, str], List[float]] = defaultdict(list)
    for (ds, noise, bit, variant, _seed), score in records.items():
        groups[(ds, noise, bit, variant)].append(score)

    rows = [
        {
            "dataset": k[0],
            "noise_rate": k[1],
            "bit": k[2],
            "variant": k[3],
            "n": len(v),
            "mean_pct": float(np.mean(v)),
            "std_pct": float(np.std(v, ddof=1)) if len(v) > 1 else None,
        }
        for k, v in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], _variant_sort_key(kv[0][3])))
    ]

    ds_map = {"MIR": "mirflickr25k_fea", "NUS": "nus_wide_tc10_fea", "COCO": "mscoco_fea"}

    def paired_diff_bit(ds: str, noise: float, bit: int, treat: str, ctrl: str) -> Optional[float]:
        keys = [(ds_map[ds], noise, bit, v, s) for s in (0, 1, 2) for v in (treat, ctrl)]
        if not all(k in records for k in keys):
            return None
        return float(np.mean([records[keys[i]] - records[keys[i + 1]] for i in range(0, 6, 2)]))

    def group_stats_bit(ds: str, noise: float, bit: int, variant: str) -> Optional[Dict[str, Any]]:
        vals = [records.get((ds_map[ds], noise, bit, variant, s)) for s in (0, 1, 2)]
        if any(v is None for v in vals):
            return None
        arr = [float(v) for v in vals if v is not None]
        return {
            "mean_pct": float(np.mean(arr)),
            "std_pct": float(np.std(arr, ddof=1)),
            "seeds_pct": arr,
        }

    def paired_diff(ds: str, noise: float, treat: str, ctrl: str) -> Optional[float]:
        return paired_diff_bit(ds, noise, 64, treat, ctrl)

    def group_stats(ds: str, noise: float, variant: str) -> Optional[Dict[str, Any]]:
        return group_stats_bit(ds, noise, 64, variant)

    # 1. off GPU 等价锚点核验
    off_parity: Dict[str, Any] = {}
    for ds_short, ds_full in ds_map.items():
        pilot_val = anchor_records.get(ds_full)
        full_val = records.get((ds_full, 0.0, 64, "off", 0))
        off_parity[ds_short] = {
            "nacg_ucch_train_off": pilot_val,
            "full198_off_s0": full_val,
            "diff_pp": (pilot_val - full_val) if (pilot_val is not None and full_val is not None) else None,
        }

    # 2. Gate P-Clean（64-bit, eta=0.0 相对主干基线变化 >= -0.10 pp）
    clean_pairs: Tuple[Tuple[str, str], ...] = tuple((v, "off") for v in mainline_variants) + (
        ("DJSRH-nacg", "DJSRH"),
        ("JDSH-nacg", "JDSH"),
    )
    gate_p_clean: Dict[str, Any] = {}
    for treat, ctrl in clean_pairs:
        diffs = {ds: paired_diff(ds, 0.0, treat, ctrl) for ds in ds_map}
        vals = list(diffs.values())
        passed = None if any(v is None for v in vals) else bool(all(float(v) >= -0.10 for v in vals if v is not None))
        gate_p_clean[f"{treat}_vs_{ctrl}"] = {
            "diffs_pp": diffs,
            "pass_ge_neg010": passed,
        }

    # 3. Gate P-Noisy Mainline（64-bit, UCCH 主线 eta=0.4）
    gate_p_noisy_mainline: Dict[str, Any] = {}
    for treat in mainline_variants:
        vs_off = {ds: paired_diff(ds, 0.4, treat, "off") for ds in ds_map}
        vs_ak1 = {ds: paired_diff(ds, 0.4, treat, "A-K1") for ds in ds_map}
        vs_jdsh = {ds: paired_diff(ds, 0.4, treat, "JDSH") for ds in ds_map}
        coco_stat = group_stats("COCO", 0.4, treat)
        vals_off = list(vs_off.values())
        avg_off = None if any(v is None for v in vals_off) else float(np.mean([v for v in vals_off if v is not None]))
        gate_pass = (
            None
            if (avg_off is None or coco_stat is None)
            else bool(all(float(v) > 0.0 for v in vals_off if v is not None) and avg_off >= 1.50 and coco_stat["mean_pct"] >= 61.8)
        )
        gate_p_noisy_mainline[treat] = {
            "vs_off_pp": vs_off,
            "avg_vs_off_pp": avg_off,
            "vs_A_K1_pp": vs_ak1,
            "vs_JDSH_pp": vs_jdsh,
            "stats": {ds: group_stats(ds, 0.4, treat) for ds in ds_map},
            "gate_pass": gate_pass,
        }

    # 4. Gate P-Noisy Plug-and-Play G2（64-bit, DJSRH-nacg / JDSH-nacg 在 eta=0.4）
    g2_gains = {
        f"{ds}/{method}": paired_diff(ds, 0.4, f"{method}-nacg", method)
        for ds in ds_map
        for method in ("DJSRH", "JDSH")
    }
    g2_vals = list(g2_gains.values())
    pos_cnt = None if any(v is None for v in g2_vals) else sum(float(v) > 0.0 for v in g2_vals if v is not None)
    avg_g2 = None if pos_cnt is None else float(np.mean([v for v in g2_vals if v is not None]))
    mir_jdsh_nacg = group_stats("MIR", 0.4, "JDSH-nacg")
    s3_comp = {
        f"{ds}/{method}-nacg_vs_{method}-pi": (
            (group_stats(ds, 0.4, f"{method}-nacg") or {}).get("mean_pct") - s3_rows_map[(ds_map[ds], 0.4, f"{method}-pi")]
            if (group_stats(ds, 0.4, f"{method}-nacg") is not None and (ds_map[ds], 0.4, f"{method}-pi") in s3_rows_map)
            else None
        )
        for ds in ds_map
        for method in ("DJSRH", "JDSH")
    }
    gate_p_noisy_g2 = {
        "gains_pp": g2_gains,
        "positive_count": pos_cnt,
        "average_gain_pp": avg_g2,
        "mir_jdsh_nacg_stats": mir_jdsh_nacg,
        "vs_s3_pi_pp": s3_comp,
        "gate_pass_5of6_avg3": (
            None
            if (pos_cnt is None or avg_g2 is None or mir_jdsh_nacg is None)
            else bool(pos_cnt >= 5 and avg_g2 >= 3.0 and mir_jdsh_nacg["std_pct"] < 1.0)
        ),
    }

    # 5. 全 6 格 64-bit 基准超越对比表与排行榜（严格锁定 bit == 64，不被 16/32/128-bit 覆盖）
    baseline_refs = ("off", "A-K1", "DJSRH", "JDSH")
    candidate_methods = mainline_variants
    cell_details: Dict[str, Any] = {}
    beats_summary: Dict[str, Dict[str, Any]] = {
        m: {"beat_count": 0, "total_available": 0, "beats_all_6": False, "avg_margin_pp": None, "margins_pp": {}}
        for m in candidate_methods
    }
    for ds in ds_map:
        for nr in (0.0, 0.4):
            ref_stats: Dict[str, Optional[float]] = {}
            for ref in baseline_refs:
                st = group_stats(ds, nr, ref)
                ref_stats[ref] = None if st is None else float(st["mean_pct"])
            avail_refs = {k: v for k, v in ref_stats.items() if v is not None}
            best_ref_name = max(avail_refs, key=lambda k: avail_refs[k]) if avail_refs else None
            best_ref_pct = avail_refs[best_ref_name] if best_ref_name is not None else None
            cell_key = f"{ds}|n{nr:g}"
            cands: Dict[str, Any] = {}
            for cand in candidate_methods:
                st = group_stats(ds, nr, cand)
                if st is None:
                    continue
                margin = (float(st["mean_pct"]) - float(best_ref_pct)) if best_ref_pct is not None else None
                beats = bool(margin > 0.0) if margin is not None else False
                beats_summary[cand]["total_available"] += 1
                if beats:
                    beats_summary[cand]["beat_count"] += 1
                beats_summary[cand]["margins_pp"][cell_key] = margin
                cands[cand] = {
                    "mean_pct": st["mean_pct"],
                    "std_pct": st["std_pct"],
                    "margin_vs_best_baseline_pp": margin,
                    "beats_all_baselines": beats,
                    "vs_each_baseline_pp": {ref: paired_diff(ds, nr, cand, ref) for ref in baseline_refs},
                }
            cell_details[cell_key] = {
                "best_baseline_name": best_ref_name,
                "best_baseline_pct": best_ref_pct,
                "baseline_means_pct": ref_stats,
                "candidates": cands,
            }
    leaderboard: List[Dict[str, Any]] = []
    for cand, info in beats_summary.items():
        info["beats_all_6"] = bool(info["total_available"] == 6 and info["beat_count"] == 6)
        valid_margins = [float(v) for v in info["margins_pp"].values() if v is not None]
        info["avg_margin_pp"] = float(np.mean(valid_margins)) if len(valid_margins) == 6 else None
        if info["total_available"] > 0:
            leaderboard.append(
                {
                    "variant": cand,
                    "beat_count": info["beat_count"],
                    "total_available": info["total_available"],
                    "beats_all_6": info["beats_all_6"],
                    "avg_margin_pp": info["avg_margin_pp"],
                }
            )
    leaderboard.sort(
        key=lambda item: (
            int(item["beat_count"]),
            float(item["avg_margin_pp"]) if item["avg_margin_pp"] is not None else float("-inf"),
            -_variant_sort_key(str(item["variant"]))[0],
        ),
        reverse=True,
    )

    # 6. 多比特表（eta=0.4, bits in {16, 32, 64, 128}，对比 off / A-K1 / DJSRH / JDSH / RCH-v6）
    multibit_table: Dict[str, Any] = {}
    for ds in ds_map:
        ds_bits: Dict[str, Any] = {}
        for bit in (16, 32, 64, 128):
            cell_methods: Dict[str, Any] = {}
            for m_name in ("off", "A-K1", "DJSRH", "JDSH", "RCH-v6"):
                st_m = group_stats_bit(ds, 0.4, bit, m_name)
                if st_m is not None:
                    cell_methods[m_name] = st_m
            v6_st = cell_methods.get("RCH-v6")
            ref_means = {r: cell_methods[r]["mean_pct"] for r in baseline_refs if r in cell_methods}
            best_r = max(ref_means.values()) if ref_means else None
            ds_bits[f"b{bit}"] = {
                "methods": cell_methods,
                "RCH_v6_vs_baselines_pp": (
                    {r: paired_diff_bit(ds, 0.4, bit, "RCH-v6", r) for r in baseline_refs}
                    if v6_st is not None
                    else None
                ),
                "RCH_v6_margin_vs_best_baseline_pp": (
                    (float(v6_st["mean_pct"]) - float(best_r)) if (v6_st is not None and best_r is not None) else None
                ),
            }
        multibit_table[ds] = ds_bits

    # 7. MIR 64-bit 噪声率鲁棒性曲线（eta in {0.0, 0.2, 0.4, 0.6}）
    mir_noise_curve: Dict[str, Any] = {}
    for nr in (0.0, 0.2, 0.4, 0.6):
        nr_methods: Dict[str, Any] = {}
        for m_name in ("off", "A-K1", "DJSRH", "JDSH", "RCH-v6"):
            st_m = group_stats_bit("MIR", nr, 64, m_name)
            if st_m is not None:
                nr_methods[m_name] = st_m
        v6_nr = nr_methods.get("RCH-v6")
        ref_nr = {r: nr_methods[r]["mean_pct"] for r in baseline_refs if r in nr_methods}
        best_nr = max(ref_nr.values()) if ref_nr else None
        mir_noise_curve[f"n{nr:g}"] = {
            "methods": nr_methods,
            "RCH_v6_vs_baselines_pp": (
                {r: paired_diff_bit("MIR", nr, 64, "RCH-v6", r) for r in baseline_refs}
                if v6_nr is not None
                else None
            ),
            "RCH_v6_margin_vs_best_baseline_pp": (
                (float(v6_nr["mean_pct"]) - float(best_nr)) if (v6_nr is not None and best_nr is not None) else None
            ),
        }

    # 8. 64-bit eta=0.4 消融阶梯与 2x2 正交单因素对照表
    ablation_variants = (
        "off",
        "RCH-v6-nog",
        "RCH-v6-fnonly",
        "RCH-v6-gonly",
        "RCH-v6-stat",
        "RCH-v6-con",
        "RCH-v6",
    )
    ablation_table: Dict[str, Any] = {}
    sfactor_2x2_table: Dict[str, Any] = {}
    has_sfactor = False
    for ds in ds_map:
        ds_ab: Dict[str, Any] = {}
        for v_ab in ablation_variants:
            st_ab = group_stats_bit(ds, 0.4, 64, v_ab)
            if st_ab is not None:
                if v_ab in ("RCH-v6-fnonly", "RCH-v6-gonly"):
                    has_sfactor = True
                ds_ab[v_ab] = {
                    **st_ab,
                    "vs_off_pp": paired_diff_bit(ds, 0.4, 64, v_ab, "off") if v_ab != "off" else 0.0,
                }
        ablation_table[ds] = ds_ab

        if any(v in ds_ab for v in ("RCH-v6-fnonly", "RCH-v6-gonly")):
            nog_m = ds_ab["RCH-v6-nog"]["mean_pct"] if "RCH-v6-nog" in ds_ab else None
            fn_m = ds_ab["RCH-v6-fnonly"]["mean_pct"] if "RCH-v6-fnonly" in ds_ab else None
            g_m = ds_ab["RCH-v6-gonly"]["mean_pct"] if "RCH-v6-gonly" in ds_ab else None
            stat_m = ds_ab["RCH-v6-stat"]["mean_pct"] if "RCH-v6-stat" in ds_ab else None
            sfactor_2x2_table[ds] = {
                "cells": {
                    k: ds_ab[k]
                    for k in ("RCH-v6-nog", "RCH-v6-fnonly", "RCH-v6-gonly", "RCH-v6-stat")
                    if k in ds_ab
                },
                "pure_graph_over_nog_pp": paired_diff_bit(ds, 0.4, 64, "RCH-v6-gonly", "RCH-v6-nog"),
                "pure_fn_over_nog_pp": paired_diff_bit(ds, 0.4, 64, "RCH-v6-fnonly", "RCH-v6-nog"),
                "fn_over_gonly_pp": paired_diff_bit(ds, 0.4, 64, "RCH-v6-stat", "RCH-v6-gonly"),
                "graph_over_fnonly_pp": paired_diff_bit(ds, 0.4, 64, "RCH-v6-stat", "RCH-v6-fnonly"),
                "joint_stat_over_nog_pp": paired_diff_bit(ds, 0.4, 64, "RCH-v6-stat", "RCH-v6-nog"),
                "interaction_pp": (
                    float((stat_m - g_m) - (fn_m - nog_m))
                    if (nog_m is not None and fn_m is not None and g_m is not None and stat_m is not None)
                    else None
                ),
            }

    pi_table = {
        (f"{k[0]}|n{k[1]:g}|{k[3]}" if k[2] == 64 else f"{k[0]}|n{k[1]:g}|b{k[2]}|{k[3]}"): v
        for k, v in sorted(pi_diag_summary.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], _variant_sort_key(kv[0][3])))
    }

    payload = {
        "rows": rows,
        "off_parity_check": off_parity,
        "Gate_P_Clean": gate_p_clean,
        "Gate_P_Noisy_Mainline": gate_p_noisy_mainline,
        "Gate_P_Noisy_PlugPlay_G2": gate_p_noisy_g2,
        "All6Cells_Beat_All_Baselines": {
            "leaderboard": leaderboard,
            "summary": beats_summary,
            "cells": cell_details,
        },
        "MultiBit_Eta04_Table": multibit_table,
        "MIR_NoiseCurve_64Bit_Table": mir_noise_curve,
        "Stepwise_Ablation_64Bit_Eta04_Table": ablation_table,
        "nacg_pi_diagnostics": pi_table,
    }
    if has_sfactor:
        payload["SingleFactor_Graph_vs_FN_2x2_Table"] = sfactor_2x2_table
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
