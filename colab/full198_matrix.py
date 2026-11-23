from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SEEDS = (0, 1, 2)
DATASETS = {
    "MIR": ("mir", "mirflickr25k_fea", "MIRFLICKR25K"),
    "NUS": ("nus", "nus_wide_tc10_fea", "NUS-WIDE-TC10"),
    "COCO": ("coco", "mscoco_fea", "MSCOCO"),
}
BATCH_FOR_DATASET = {"MIR": "B2", "NUS": "B3", "COCO": "B4"}
EXTERNAL_BATCH_FOR_DATASET = {"MIR": "B5", "NUS": "B6", "COCO": "B6"}
P2 = "P2-OURS-MAIN-ALLBIT"
P3 = "P3-OURS-ROBUST64"
P4 = "P4-EXTERNAL-CORE64"
P5 = "P5-EXTERNAL-ALLBIT-OPTIONAL"


@dataclass(frozen=True)
class RunSpec:
    run_key: str
    scope: str
    packages: tuple[str, ...]
    batch: str
    variant: str
    method_class: str
    dataset: str
    data_name: str
    data_subdir: str
    bit: int
    noise_rate: float
    noise_seed: int
    seed: int


def make_spec(dataset: str, variant: str, bit: int, noise_rate: float, seed: int) -> RunSpec:
    ds_key, data_name, data_subdir = DATASETS[dataset]
    if variant in ("off", "A-K1"):
        method_class = "internal"
        batch = BATCH_FOR_DATASET[dataset]
        packages = (P2,) if noise_rate == 0.4 else (P3,)
        if bit == 64 and noise_rate == 0.4:
            packages = (P2, P3)
        scope = "CORE"
    else:
        method_class = "external"
        if bit == 64:
            packages = (P4,)
            batch = EXTERNAL_BATCH_FOR_DATASET[dataset]
            scope = "CORE"
        else:
            packages = (P5,)
            batch = "B7"
            scope = "OPTIONAL"
    return RunSpec(
        run_key=f"{data_name}_n{noise_rate:g}_b{bit}_s{seed}_{variant}",
        scope=scope,
        packages=packages,
        batch=batch,
        variant=variant,
        method_class=method_class,
        dataset=dataset,
        data_name=data_name,
        data_subdir=data_subdir,
        bit=bit,
        noise_rate=noise_rate,
        noise_seed=0,
        seed=seed,
    )


def build_specs() -> list[RunSpec]:
    specs = []
    for dataset in ("MIR", "NUS", "COCO"):
        for variant in ("off", "A-K1"):
            bits = (32, 64, 128) if dataset == "MIR" else (16, 32, 64, 128)
            for bit in bits:
                for seed in SEEDS:
                    specs.append(make_spec(dataset, variant, bit, 0.4, seed))
            extra_rates = (0.0, 0.2, 0.6) if dataset == "MIR" else (0.0,)
            for noise_rate in extra_rates:
                for seed in SEEDS:
                    specs.append(make_spec(dataset, variant, 64, noise_rate, seed))
    for dataset in ("MIR", "NUS", "COCO"):
        for variant in ("DJSRH", "JDSH"):
            rates = (0.0, 0.2, 0.4, 0.6) if dataset == "MIR" else (0.0, 0.4)
            for noise_rate in rates:
                for seed in SEEDS:
                    specs.append(make_spec(dataset, variant, 64, noise_rate, seed))
            for bit in (16, 32, 128):
                for seed in SEEDS:
                    specs.append(make_spec(dataset, variant, bit, 0.4, seed))
    keys = [spec.run_key for spec in specs]
    if len(specs) != 198 or len(set(keys)) != 198:
        raise AssertionError(f"198组清单生成错误: runs={len(specs)}, unique={len(set(keys))}")
    return specs


def build_command(
    spec: RunSpec,
    baseline_dir: Path,
    data_root: Path,
    out_dir: Path,
    python_bin: str = "python3",
    manifest_root: Path | None = None,
    num_workers: int = 5,
) -> list[str]:
    if spec.method_class == "internal":
        return [
            python_bin, str(baseline_dir / "ours/i1_train.py"),
            "--seed", str(spec.seed), "--data_root", str(data_root), "--out", str(out_dir),
            "--log_name", "I1", "--pretrain_dir", "I1",
            "--noise_rate", str(spec.noise_rate), "--noise_seed", str(spec.noise_seed),
            "--variant", spec.variant, "--num_views", "4", "--soft_beta", "0.5", "--i2_lambda", "1.0",
            "--mcr_signal", "consistency", "--mcr_a_warmup", "1",
            "--mcr_gamma", "0", "--mcr_delta", "0.1", "--mcr_normalize", "n",
            "--mcr_rho_bar", "0.8", "--mcr_seed_offset", "1000",
            "--probe_seed_offset", "100", "--randq_seed_offset", "200", "--perm_seed_offset", "300",
            "--data_name", spec.data_name, "--bit", str(spec.bit), "--alpha", "0.7",
            "--num_hiden_layers", "3", "2", "--margin", "0.2", "--max_epochs", "20",
            "--train_batch_size", "256", "--eval_batch_size", "256", "--shift", "0.1",
            "--lr", "0.0001", "--wd", "0.000001", "--optimizer", "Adam", "--ls", "linear",
            "--warmup_epoch", "1", "--K", "4096", "--T", "0.9", "--momentum", "0.4",
            "--num_workers", str(num_workers), "--no_save_ckpt",
        ]
    command = [
        python_bin, str(baseline_dir / "baselines/run_external.py"),
        "--method", spec.variant, "--data_root", str(data_root), "--data_name", spec.data_name,
        "--bit", str(spec.bit), "--seed", str(spec.seed),
        "--noise_rate", str(spec.noise_rate), "--noise_seed", str(spec.noise_seed),
        "--out", str(out_dir), "--max_epochs", "20",
        "--train_batch_size", "256", "--eval_batch_size", "256",
        "--num_workers", str(num_workers), "--require_cuda", "--no_save_ckpt",
    ]
    if manifest_root is not None:
        command.extend(["--data_manifest", str(manifest_root / f"{spec.data_name}.json")])
    return command


def spec_payload(spec: RunSpec) -> dict[str, Any]:
    return asdict(spec)


def validate_summary(spec: RunSpec, summary_path: Path, require_train_log: bool = True) -> dict[str, Any]:
    run_dir = summary_path.parent
    required = [run_dir / "noise_mask.npy", run_dir / "noise_sources.npy", run_dir / "label_inconsistency.json"]
    if require_train_log:
        required.append(run_dir / "train.log")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"{spec.run_key}缺少产物: {missing}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    parsed = data["parsed_args"]
    checks = {
        "variant": data.get("variant") == spec.variant,
        "seed": int(data.get("seed", -1)) == spec.seed,
        "data_name": parsed.get("data_name") == spec.data_name,
        "bit": int(parsed.get("bit", -1)) == spec.bit,
        "noise_rate": math.isclose(float(data["noise_rate"]), spec.noise_rate, abs_tol=1e-12),
        "noise_seed": int(data.get("noise_seed", -1)) == spec.noise_seed,
        "max_epochs": int(parsed.get("max_epochs", -1)) == 20,
        "epoch_records": len(data.get("epoch_records", [])) == 20,
        "best_val_epoch": 0 <= int(data.get("best_val_epoch", -1)) < 20,
        "test_sizes": int(data.get("n_query", 0)) > 0 and int(data.get("n_db", 0)) > 0,
    }
    for metric in ("test_i2t_official", "test_t2i_official", "test_avg_official", "test_i2t_stable", "test_t2i_stable", "test_avg_stable"):
        value = float(data[metric])
        checks[metric] = math.isfinite(value) and 0.0 <= value <= 1.0
    if spec.method_class == "external":
        hyper = data.get("external_hyperparams", {})
        checks["source_commit"] = bool(hyper.get("source", {}).get("commit"))
        checks["data_manifest"] = bool(data.get("data_manifest")) and bool(data.get("data_manifest_sha256"))
        checks["trainable_parameters"] = int(data.get("trainable_parameters_total", 0)) > 0
        checks["peak_cuda_memory"] = int(data.get("peak_cuda_memory_bytes", -1)) >= 0
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{spec.run_key}摘要核验失败: {failed}")
    return data


def select_specs(specs: Iterable[RunSpec], batches: set[str] | None = None, packages: set[str] | None = None) -> list[RunSpec]:
    selected = []
    for spec in specs:
        if batches and spec.batch not in batches:
            continue
        if packages and not packages.intersection(spec.packages):
            continue
        selected.append(spec)
    return selected
