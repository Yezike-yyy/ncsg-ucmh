"""M5-RCHV6-FINAL51（51 组旗舰完整实验矩阵：多比特 + 噪声鲁棒曲线 + 逐步消融）执行器。

包含三大正交实验包（共 51 组唯一运行，与 Iter-6 的 36 组 64-bit 主表运行键零重叠）：
  1. 多比特评测（27 组）：RCH-v6 (alpha=0.65, lambda_graph=0.35, sharpen=2.5)，MIR / NUS / COCO × bits {16, 32, 128} × eta=0.4 × seeds {0, 1, 2}；
  2. 噪声率曲线（6 组）：RCH-v6 (alpha=0.65, lambda_graph=0.35, sharpen=2.5)，MIR × 64-bit × eta {0.2, 0.6} × seeds {0, 1, 2}；
  3. 逐步消融（18 组）：RCH-v6-nog (lambda_graph=0.0, alpha=0.65, sharpen=2.5) 与 RCH-v6-stat (lambda_graph=0.35, alpha=0.65, sharpen=1.0)，
     MIR / NUS / COCO × 64-bit × eta=0.4 × seeds {0, 1, 2}。
默认 dry-run 只打印命令行与运行清单而不启动训练；传入 ``--execute`` 时以 ``--workers 4``（小显存 GPU 自动限流告警）并发训练并原子归档至
独立云盘目录（默认 ``/content/drive/MyDrive/NACG_final51_runs``），自动注入 ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``。
若任一运行失败，主进程严格以非零状态码 ``sys.exit(1)`` 退出。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent
for _p in (BASELINE, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from colab.full198_matrix import DATASETS  # type: ignore[import-not-found]
from colab.run_full198 import archive_success, stage_dataset  # type: ignore[import-not-found]

MIN_FULL_WORKERS_VRAM_BYTES = 32 * (1024 ** 3)


@dataclass(frozen=True)
class NACGFinal51Spec:
    run_key: str
    group: str
    dataset: str
    data_name: str
    data_subdir: str
    variant: str
    bit: int
    noise_rate: float
    seed: int
    alpha: float
    lambda_graph: float
    rch_v6_bridge_sharpen: float


def build_final51_specs() -> List[NACGFinal51Spec]:
    specs: List[NACGFinal51Spec] = []

    # 1) 多比特主评测（27 组）：数据集按 MIR -> NUS -> COCO 排序使轻量作业优先产出心跳
    for ds_key in ("MIR", "NUS", "COCO"):
        _, data_name, data_subdir = DATASETS[ds_key]
        for bit in (16, 32, 128):
            for seed in (0, 1, 2):
                specs.append(
                    NACGFinal51Spec(
                        run_key=f"{data_name}_n0.4_b{bit}_s{seed}_RCH-v6",
                        group="multibit",
                        dataset=ds_key,
                        data_name=data_name,
                        data_subdir=data_subdir,
                        variant="RCH-v6",
                        bit=bit,
                        noise_rate=0.4,
                        seed=seed,
                        alpha=0.65,
                        lambda_graph=0.35,
                        rch_v6_bridge_sharpen=2.5,
                    )
                )

    # 2) MIR 64-bit 噪声率鲁棒性曲线补点（6 组：eta=0.2, 0.6；联立 Iter-6 的 eta=0.0, 0.4 构成完整 4 点曲线）
    _, mir_name, mir_subdir = DATASETS["MIR"]
    for noise_rate in (0.2, 0.6):
        for seed in (0, 1, 2):
            specs.append(
                NACGFinal51Spec(
                    run_key=f"{mir_name}_n{noise_rate:g}_b64_s{seed}_RCH-v6",
                    group="noisecurve",
                    dataset="MIR",
                    data_name=mir_name,
                    data_subdir=mir_subdir,
                    variant="RCH-v6",
                    bit=64,
                    noise_rate=noise_rate,
                    seed=seed,
                    alpha=0.65,
                    lambda_graph=0.35,
                    rch_v6_bridge_sharpen=2.5,
                )
            )

    # 3) 64-bit eta=0.4 逐步消融阶梯（18 组：RCH-v6-nog 与 RCH-v6-stat）
    for ds_key in ("MIR", "NUS", "COCO"):
        _, data_name, data_subdir = DATASETS[ds_key]
        for variant, lam, alpha, sharpen in (
            ("RCH-v6-nog", 0.0, 0.65, 2.5),
            ("RCH-v6-stat", 0.35, 0.65, 1.0),
        ):
            for seed in (0, 1, 2):
                specs.append(
                    NACGFinal51Spec(
                        run_key=f"{data_name}_n0.4_b64_s{seed}_{variant}",
                        group="ablation",
                        dataset=ds_key,
                        data_name=data_name,
                        data_subdir=data_subdir,
                        variant=variant,
                        bit=64,
                        noise_rate=0.4,
                        seed=seed,
                        alpha=alpha,
                        lambda_graph=lam,
                        rch_v6_bridge_sharpen=sharpen,
                    )
                )

    if len(specs) != 51 or len({s.run_key for s in specs}) != 51:
        raise AssertionError(f"Final-51 清单必须为 51 组唯一键，当前={len(specs)}")
    return specs


def build_final51_command(
    spec: NACGFinal51Spec,
    baseline_dir: Path,
    local_data_root: Path,
    out_dir: Path,
    python_bin: str = "python3",
    num_workers: int = 0,
) -> List[str]:
    return [
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
        "--alpha", str(spec.alpha),
        "--lambda_graph", str(spec.lambda_graph),
        "--rch_v6_bridge_sharpen", str(spec.rch_v6_bridge_sharpen),
        "--num_workers", str(num_workers),
        "--require_cuda",
        "--no_save_ckpt",
    ]


from colab.summary_validator import validate_nacg_run_dir  # type: ignore[import-not-found]


def is_completed_valid(run_dir: Path, spec: NACGFinal51Spec) -> bool:
    """严格校验已完成或待跳过归档目录的摘要字段、超参（含 alpha、lambda_graph、rch_v6_bridge_sharpen）、轮数与产物。"""
    return validate_nacg_run_dir(
        run_dir,
        expected_variant=spec.variant,
        expected_seed=spec.seed,
        expected_noise_rate=spec.noise_rate,
        expected_bit=spec.bit,
        expected_data_name=spec.data_name,
        expected_lambda_graph=spec.lambda_graph,
        expected_alpha=spec.alpha,
        expected_bridge_sharpen=spec.rch_v6_bridge_sharpen,
    )


def resolve_safe_workers(requested_workers: int, total_vram_bytes: Optional[int] = None) -> int:
    """当 GPU 总显存 < 32 GiB 且请求并发数 > 2 时自动限流并打印显式告警，防止小显存 GPU OOM。"""
    vram = total_vram_bytes
    if vram is None:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                vram = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            vram = None

    if vram is not None and vram < MIN_FULL_WORKERS_VRAM_BYTES and requested_workers > 2:
        gib = float(vram) / float(1024 ** 3)
        clamped = 1 if gib < 16.0 else 2
        print(
            json.dumps(
                {
                    "warning": "gpu_vram_below_32gib_clamping_workers",
                    "gpu_total_gib": round(gib, 2),
                    "requested_workers": int(requested_workers),
                    "effective_workers": int(clamped),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return clamped
    return int(requested_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", default="/content/baseline")
    parser.add_argument("--data_root", default="/content/drive/MyDrive/UCCH_data")
    parser.add_argument("--local_data_root", default="/content/data")
    parser.add_argument("--local_root", default="/content/NACG_final51_local")
    parser.add_argument("--out_root", default="/content/drive/MyDrive/NACG_final51_runs")
    parser.add_argument("--python_bin", default="python3")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.shard_count < 1 or not 0 <= args.shard < args.shard_count:
        raise ValueError("并发和分片参数无效")

    all_specs = build_final51_specs()
    if args.groups:
        allowed_groups = set(args.groups)
        all_specs = [s for s in all_specs if s.group in allowed_groups]
    specs = [s for i, s in enumerate(all_specs) if i % args.shard_count == args.shard]
    baseline, source, local_data, local, output = map(
        Path,
        (args.baseline_dir, args.data_root, args.local_data_root, args.local_root, args.out_root),
    )

    if not args.execute or args.dry_run:
        for spec in specs:
            print(
                shlex.join(
                    build_final51_command(
                        spec,
                        baseline,
                        local_data,
                        local / spec.run_key,
                        args.python_bin,
                        args.num_workers,
                    )
                )
            )
        print(json.dumps({"dry_run": True, "runs": len(specs), "executed": False}))
        return

    effective_workers = resolve_safe_workers(args.workers)

    for spec in {s.dataset: s for s in specs}.values():
        stage_dataset(source / spec.data_subdir, local_data / spec.data_subdir)
    output.mkdir(parents=True, exist_ok=True)

    def execute_one(spec: NACGFinal51Spec) -> str:
        try:
            destination = output / spec.run_key
            if (destination / "summary.json").is_file():
                if is_completed_valid(destination, spec):
                    return f"SKIPPED {spec.run_key}"
                raise ValueError(f"已有摘要不匹配或不完整: {destination}")

            attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            run_dir = local / spec.run_key / attempt
            run_dir.mkdir(parents=True, exist_ok=False)
            cmd = build_final51_command(spec, baseline, local_data, run_dir, args.python_bin, args.num_workers)
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env.setdefault(key, "1")
            with (run_dir / "train.log").open("w", encoding="utf-8") as log:
                res = subprocess.run(cmd, cwd=baseline, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if res.returncode != 0 or not is_completed_valid(run_dir, spec):
                raise RuntimeError(f"子进程失败或产物校验未通过 exit={res.returncode}; 日志={run_dir / 'train.log'}")
            archive_success(run_dir, destination)
            if not is_completed_valid(destination, spec):
                raise RuntimeError(f"归档目录产物校验未通过: {destination}")
            return f"COMPLETED {spec.run_key}"
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            try:
                failures = output / "_failed"
                failures.mkdir(parents=True, exist_ok=True)
                with (failures / f"{spec.run_key}.json").open("w", encoding="utf-8") as handle:
                    json.dump({"run_key": spec.run_key, "error": str(exc)}, handle, ensure_ascii=False)
            except OSError as write_exc:
                return f"FAILED {spec.run_key}: {exc}; 写失败记录失败: {write_exc}"
            return f"FAILED {spec.run_key}: {exc}"

    failed: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
        for status_msg in pool.map(execute_one, specs):
            print(status_msg, flush=True)
            if status_msg.startswith("FAILED"):
                failed.append(status_msg)

    if failed:
        print(
            json.dumps({"status": "FAILED", "failed_count": len(failed), "total": len(specs)}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
