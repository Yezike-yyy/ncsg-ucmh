from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent.parent
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from colab.full198_matrix import RunSpec, build_command, select_specs, spec_payload, validate_summary


def load_specs(path: Path) -> list[RunSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for item in payload["runs"]:
        values = {key: value for key, value in item.items() if key != "command"}
        values["packages"] = tuple(values["packages"])
        specs.append(RunSpec(**values))
    if len(specs) != 198 or len({spec.run_key for spec in specs}) != 198:
        raise ValueError("计划必须包含198个唯一run_key")
    return specs


def files_match(source: Path, destination: Path) -> bool:
    if not source.is_dir() or not destination.is_dir():
        return False
    source_files = [path for path in source.rglob("*") if path.is_file()]
    if not source_files:
        return False
    for path in source_files:
        target = destination / path.relative_to(source)
        if not target.is_file() or target.stat().st_size != path.stat().st_size:
            return False
    return True


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def stage_dataset(source: Path, destination: Path) -> None:
    lock_path = destination.parent / f".{destination.name}.stage.lock"
    with exclusive_lock(lock_path):
        if files_match(source, destination):
            print(json.dumps({"data_cache": "reused", "path": str(destination)}, ensure_ascii=False), flush=True)
            return
        if not source.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {source}")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        if not files_match(source, destination):
            raise RuntimeError(f"数据缓存核验失败: {destination}")
        print(json.dumps({"data_cache": "staged", "path": str(destination)}, ensure_ascii=False), flush=True)


def ensure_data_manifest(local_data_root: Path, manifest_root: Path, data_name: str) -> Path:
    from baselines.protocol import build_data_manifest, write_json_atomic

    manifest_path = manifest_root / f"{data_name}.json"
    with exclusive_lock(manifest_root / f".{data_name}.manifest.lock"):
        if not manifest_path.is_file():
            write_json_atomic(manifest_path, build_data_manifest(local_data_root, data_name))
            print(json.dumps({"data_manifest": str(manifest_path), "created": True}, ensure_ascii=False), flush=True)
    return manifest_path


def archive_success(attempt_dir: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    optional_files = ["run_config.json", "i1_buffers_latest.npz", "test_codes.npz", "best_checkpoint.pt"]
    for name in ["noise_mask.npy", "noise_sources.npy", "label_inconsistency.json", "train.log", *optional_files]:
        source = attempt_dir / name
        if source.is_file():
            shutil.copyfile(source, drive_dir / name)
    if (attempt_dir / "logs").is_dir():
        shutil.copytree(attempt_dir / "logs", drive_dir / "logs", dirs_exist_ok=True)
    shutil.copyfile(attempt_dir / "summary.json", drive_dir / "summary.pending.json")
    os.replace(drive_dir / "summary.pending.json", drive_dir / "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行或打印198组计划；默认只打印，只有--execute才训练")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--baseline_dir", default="/content/baseline")
    parser.add_argument("--data_root", default="/content/drive/MyDrive/UCCH_data")
    parser.add_argument("--local_data_root", default="/content/data")
    parser.add_argument("--local_root", default="/content/FULL198_local")
    parser.add_argument("--out_root", default="/content/drive/MyDrive/FULL198_runs")
    parser.add_argument("--manifest_root", default="/content/full198_manifests")
    parser.add_argument("--python_bin", default="python3")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--batches", nargs="*", default=[])
    parser.add_argument("--packages", nargs="*", default=[])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard < args.shard_count:
        raise ValueError("shard_count必须为正且0<=shard<shard_count")
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    source_data_root = Path(args.data_root).expanduser().resolve()
    local_data_root = Path(args.local_data_root).expanduser().resolve()
    local_root = Path(args.local_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    selected = select_specs(load_specs(Path(args.plan).expanduser().resolve(),), set(args.batches), set(args.packages))
    selected = [spec for index, spec in enumerate(selected) if index % args.shard_count == args.shard]
    if not selected:
        raise ValueError("当前过滤和分片没有运行项")
    if not args.execute:
        for spec in selected:
            command = build_command(
                spec, baseline_dir, local_data_root, local_root / spec.run_key,
                python_bin=args.python_bin, manifest_root=manifest_root, num_workers=args.num_workers,
            )
            print(shlex.join(command))
        print(json.dumps({"dry_run": True, "runs": len(selected), "executed": False}, ensure_ascii=False))
        return

    active_datasets = {spec.dataset: spec for spec in selected}
    for dataset, spec in active_datasets.items():
        stage_dataset(source_data_root / spec.data_subdir, local_data_root / spec.data_subdir)
    external_data_names = sorted({spec.data_name for spec in selected if spec.method_class == "external"})
    if external_data_names:
        manifest_root.mkdir(parents=True, exist_ok=True)
        for data_name in external_data_names:
            ensure_data_manifest(local_data_root, manifest_root, data_name)

    out_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for index, spec in enumerate(selected, start=1):
        drive_dir = out_root / spec.run_key
        if (drive_dir / "summary.json").is_file():
            validate_summary(spec, drive_dir / "summary.json")
            completed += 1
            print(json.dumps({"status": "SKIPPED_VERIFIED", "run_key": spec.run_key, "completed": completed, "selected": len(selected)}, ensure_ascii=False), flush=True)
            continue
        if drive_dir.exists():
            raise RuntimeError(f"输出目录存在但无有效summary，保留现场并停止: {drive_dir}")
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_p{os.getpid()}"
        attempt_dir = local_root / spec.run_key / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        command = build_command(
            spec, baseline_dir, local_data_root, attempt_dir,
            python_bin=args.python_bin, manifest_root=manifest_root, num_workers=args.num_workers,
        )
        print(json.dumps({"status": "STARTED", "index": index, "selected": len(selected), "batch": spec.batch, "run_key": spec.run_key, "command": command}, ensure_ascii=False), flush=True)
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(thread_key, "1")
        with (attempt_dir / "train.log").open("w", encoding="utf-8", buffering=1) as log:
            result = subprocess.run(command, cwd=str(baseline_dir), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not (attempt_dir / "summary.json").is_file():
            failure_dir = out_root / "_failed" / spec.run_key
            failure_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(attempt_dir / "train.log", failure_dir / f"{attempt}.log")
            (failure_dir / f"{attempt}.json").write_text(json.dumps({
                "run_key": spec.run_key, "attempt": attempt, "returncode": result.returncode,
                "command": command, "local_evidence": str(attempt_dir), "status": "FAILED_NOT_RETRIED",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"运行失败，已保留日志且未自动重试: {spec.run_key}; exit={result.returncode}")
        validate_summary(spec, attempt_dir / "summary.json")
        archive_success(attempt_dir, drive_dir)
        validate_summary(spec, drive_dir / "summary.json")
        completed += 1
        print(json.dumps({"status": "COMPLETED_VERIFIED", "run_key": spec.run_key, "completed": completed, "selected": len(selected)}, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "BATCH_COMPLETE", "completed": completed, "selected": len(selected), "batches": sorted(set(args.batches)), "shard": args.shard, "shard_count": args.shard_count}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
