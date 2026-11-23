from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io as sio

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    torch = None  # type: ignore[assignment]

from ours.label_diagnostics import compute_label_inconsistency
from ours.mcr import build_correspondence_permutation

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "mirflickr25k_fea": {
        "dataset": "MIR",
        "subdir": "MIRFLICKR25K",
        "test_size": 2000,
        "files": [
            ("mirflickr25k-iall-vgg.mat", "XAll", "scipy"),
            ("mirflickr25k-yall.mat", "YAll", "scipy"),
            ("mirflickr25k-lall.mat", "LAll", "scipy"),
        ],
    },
    "nus_wide_tc10_fea": {
        "dataset": "NUS",
        "subdir": "NUS-WIDE-TC10",
        "test_size": 2100,
        "files": [
            ("nus-wide-tc10-xall-vgg.mat", "XAll", "scipy"),
            ("nus-wide-tc10-yall.mat", "YAll", "scipy"),
            ("nus-wide-tc10-lall.mat", "LAll", "scipy"),
        ],
    },
    "mscoco_fea": {
        "dataset": "COCO",
        "subdir": "MSCOCO",
        "test_size": 5000,
        "files": [("MSCOCO_deep_doc2vec_data_rand.h5py", None, "h5py")],
    },
}


@dataclass
class ProtocolData:
    data_name: str
    images: np.ndarray
    texts: np.ndarray
    labels: np.ndarray
    query_images: np.ndarray
    query_texts: np.ndarray
    query_labels: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class NoiseData:
    train_texts: np.ndarray
    mask: np.ndarray
    sources: np.ndarray
    label_diagnostic: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_protocol_data(data_root: Path, data_name: str) -> ProtocolData:
    if data_name not in DATASET_SPECS:
        raise ValueError(f"未知数据集: {data_name}")
    spec = DATASET_SPECS[data_name]
    root = data_root / spec["subdir"]
    arrays: dict[str, np.ndarray] = {}
    for filename, key, loader in spec["files"]:
        path = root / filename
        if loader == "scipy":
            with path.open("rb") as handle:
                arrays[key] = np.asarray(sio.loadmat(handle)[key])
        else:
            with h5py.File(path, "r") as handle:
                for h5_key in ("XAll", "YAll", "LAll"):
                    arrays[h5_key] = np.asarray(handle[h5_key][()])
    images = np.ascontiguousarray(arrays["XAll"], dtype=np.float32)
    texts = np.ascontiguousarray(arrays["YAll"], dtype=np.float32)
    labels = np.ascontiguousarray(arrays["LAll"])
    if images.ndim != 2 or texts.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"{data_name}的图像、文本和标签必须都是二维数组")
    if not (len(images) == len(texts) == len(labels)):
        raise ValueError(f"{data_name}的模态样本数不一致")
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError(f"{data_name}包含非有限特征")
    test_size = int(spec["test_size"])
    if len(images) <= test_size:
        raise ValueError(f"{data_name}样本数不足")
    return ProtocolData(
        data_name=data_name,
        images=images[:-test_size],
        texts=texts[:-test_size],
        labels=labels[:-test_size],
        query_images=images[-test_size:],
        query_texts=texts[-test_size:],
        query_labels=labels[-test_size:],
    )


def apply_training_noise(data: ProtocolData, noise_rate: float, noise_seed: int) -> NoiseData:
    mask, sources = build_correspondence_permutation(data.n_train, noise_rate, noise_seed)
    noisy_texts = data.texts.copy()
    noisy_texts[mask] = data.texts[sources[mask]]
    _, diagnostic = compute_label_inconsistency(data.labels, mask, sources)
    diagnostic["text_permutation_rate_eta"] = float(noise_rate)
    diagnostic["noise_seed"] = int(noise_seed)
    return NoiseData(noisy_texts, mask, sources, diagnostic)


def save_noise_data(out_dir: Path, noise: NoiseData) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "noise_mask.npy"
    sources_path = out_dir / "noise_sources.npy"
    diagnostic_path = out_dir / "label_inconsistency.json"
    np.save(mask_path, noise.mask)
    np.save(sources_path, noise.sources)
    write_json_atomic(diagnostic_path, noise.label_diagnostic)
    return {
        "noise_mask_sha256": sha256_file(mask_path),
        "noise_sources_sha256": sha256_file(sources_path),
        "label_inconsistency_sha256": sha256_file(diagnostic_path),
    }


def build_data_manifest(data_root: Path, data_name: str) -> dict[str, Any]:
    spec = DATASET_SPECS[data_name]
    root = data_root / spec["subdir"]
    files = []
    for filename, key, loader in spec["files"]:
        path = root / filename
        files.append({
            "path": f"{spec['subdir']}/{filename}",
            "key": key,
            "loader": loader,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": "1.0",
        "protocol": "UCCH_PREEXTRACTED_FEATURES_FIXED_SUFFIX_QUERY",
        "data_name": data_name,
        "dataset": spec["dataset"],
        "test_size": spec["test_size"],
        "files": files,
    }


def _map_values_from_order_cpu_ref(
    order_all: np.ndarray,
    retrieval_labels: np.ndarray,
    query_labels: np.ndarray,
) -> list[float]:
    ordered_labels = retrieval_labels[order_all]
    values = []
    for index in range(order_all.shape[0]):
        relevant = np.dot(ordered_labels[index], query_labels[index]) > 0
        total_positive = int(relevant.sum())
        if total_positive == 0:
            continue
        precision = relevant.cumsum() / np.arange(1.0, 1.0 + relevant.shape[0])
        values.append(float(np.dot(relevant, precision) / total_positive))
    return values


_map_values_from_order = _map_values_from_order_cpu_ref


def _map_values_from_relevant(
    ordered_relevant: np.ndarray,
    arange_vec: np.ndarray,
) -> list[float]:
    values = []
    for index in range(ordered_relevant.shape[0]):
        relevant = ordered_relevant[index]
        total_positive = int(relevant.sum())
        if total_positive == 0:
            continue
        precision = relevant.cumsum() / arange_vec
        values.append(float(np.dot(relevant, precision) / total_positive))
    return values


def _map_direction_cpu_ref(
    retrieval_codes: np.ndarray,
    query_codes: np.ndarray,
    retrieval_labels: np.ndarray,
    query_labels: np.ndarray,
    chunk_size: int,
    call_index: int,
) -> dict[str, Any]:
    import scipy.spatial.distance

    official_values = []
    stable_values = []
    n_db = int(len(retrieval_labels))
    for start in range(0, len(query_labels), chunk_size):
        end = min(start + chunk_size, len(query_labels))
        distances = scipy.spatial.distance.cdist(query_codes[start:end], retrieval_codes, metric="hamming")
        official_order = distances.argsort()
        stable_order = distances.argsort(kind="mergesort")
        del distances
        official_values.extend(_map_values_from_order_cpu_ref(official_order, retrieval_labels, query_labels[start:end]))
        stable_values.extend(_map_values_from_order_cpu_ref(stable_order, retrieval_labels, query_labels[start:end]))
    official = float(np.mean(official_values)) if official_values else 0.0
    stable = float(np.mean(stable_values)) if stable_values else 0.0
    return {
        "call_index": call_index,
        "metric": "hamming",
        "n_query": int(len(query_labels)),
        "n_db": n_db,
        "map_official": official,
        "map_stable": stable,
        "query_chunk_size": chunk_size,
        "query_chunks": (len(query_labels) + chunk_size - 1) // chunk_size,
    }


def _map_direction(
    retrieval_codes: np.ndarray,
    query_codes: np.ndarray,
    retrieval_labels: np.ndarray,
    query_labels: np.ndarray,
    chunk_size: int,
    call_index: int,
) -> dict[str, Any]:
    if (
        torch is None
        or not torch.cuda.is_available()
        or os.environ.get("DISABLE_PROTOCOL_GPU_MAP") == "1"
        or len(retrieval_labels) == 0
        or len(query_labels) == 0
        or retrieval_codes.shape[1] == 0
    ):
        return _map_direction_cpu_ref(
            retrieval_codes, query_codes, retrieval_labels, query_labels, chunk_size, call_index,
        )

    device = torch.device("cuda")
    n_db = int(len(retrieval_labels))
    n_bits = float(retrieval_codes.shape[1])
    with torch.no_grad():
        r_codes_gpu = torch.as_tensor(retrieval_codes, device=device, dtype=torch.float64)
        q_codes_gpu = torch.as_tensor(query_codes, device=device, dtype=torch.float64)
        is_pm1 = bool(
            torch.all((r_codes_gpu == 1.0) | (r_codes_gpu == -1.0)).item()
            and torch.all((q_codes_gpu == 1.0) | (q_codes_gpu == -1.0)).item()
        )
        r_p = r_n = r_z = None
        if not is_pm1:
            is_m101 = bool(
                torch.all((r_codes_gpu == 1.0) | (r_codes_gpu == -1.0) | (r_codes_gpu == 0.0)).item()
                and torch.all((q_codes_gpu == 1.0) | (q_codes_gpu == -1.0) | (q_codes_gpu == 0.0)).item()
            )
            if not is_m101:
                del r_codes_gpu, q_codes_gpu
                return _map_direction_cpu_ref(
                    retrieval_codes, query_codes, retrieval_labels, query_labels, chunk_size, call_index,
                )
            r_p = (r_codes_gpu == 1.0).to(torch.float64)
            r_n = (r_codes_gpu == -1.0).to(torch.float64)
            r_z = (r_codes_gpu == 0.0).to(torch.float64)
            del r_codes_gpu

        r_lab_gpu = torch.as_tensor(retrieval_labels, device=device, dtype=torch.float64)
        q_lab_gpu = torch.as_tensor(query_labels, device=device, dtype=torch.float64)
        gpu_chunk = max(chunk_size, 512)
        arange_vec = np.arange(1.0, 1.0 + n_db)
        official_values: list[float] = []
        stable_values: list[float] = []

        for start in range(0, len(query_labels), gpu_chunk):
            end = min(start + gpu_chunk, len(query_labels))
            qc = q_codes_gpu[start:end]
            if is_pm1:
                dist_gpu = ((n_bits - (qc @ r_codes_gpu.t())) * 0.5) / n_bits
            else:
                assert r_p is not None and r_n is not None and r_z is not None
                matches = (
                    ((qc == 1.0).to(torch.float64) @ r_p.t())
                    + ((qc == -1.0).to(torch.float64) @ r_n.t())
                    + ((qc == 0.0).to(torch.float64) @ r_z.t())
                )
                dist_gpu = (n_bits - matches) / n_bits
            distances = dist_gpu.cpu().numpy()
            stable_order_gpu = torch.argsort(dist_gpu, dim=1, stable=True)
            del dist_gpu
            official_order = distances.argsort()
            del distances

            rel_mask_gpu = (q_lab_gpu[start:end] @ r_lab_gpu.t()) > 0
            rel_official = torch.gather(
                rel_mask_gpu, 1, torch.from_numpy(official_order).to(device=device),
            ).cpu().numpy()
            rel_stable = torch.gather(
                rel_mask_gpu, 1, stable_order_gpu,
            ).cpu().numpy()
            del rel_mask_gpu, official_order, stable_order_gpu

            official_values.extend(_map_values_from_relevant(rel_official, arange_vec))
            stable_values.extend(_map_values_from_relevant(rel_stable, arange_vec))

    official = float(np.mean(official_values)) if official_values else 0.0
    stable = float(np.mean(stable_values)) if stable_values else 0.0
    return {
        "call_index": call_index,
        "metric": "hamming",
        "n_query": int(len(query_labels)),
        "n_db": n_db,
        "map_official": official,
        "map_stable": stable,
        "query_chunk_size": chunk_size,
        "query_chunks": (len(query_labels) + chunk_size - 1) // chunk_size,
    }


def evaluate_codes(
    retrieval_images: np.ndarray,
    retrieval_texts: np.ndarray,
    retrieval_labels: np.ndarray,
    query_images: np.ndarray,
    query_texts: np.ndarray,
    query_labels: np.ndarray,
    query_chunk_size: int = 64,
) -> dict[str, Any]:
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size必须为正整数")
    records = [
        _map_direction(retrieval_texts, query_images, retrieval_labels, query_labels, query_chunk_size, 0),
        _map_direction(retrieval_images, query_texts, retrieval_labels, query_labels, query_chunk_size, 1),
    ]
    i2t = records[0]["map_official"]
    t2i = records[1]["map_official"]
    return {
        "i2t_official": float(i2t),
        "t2i_official": float(t2i),
        "avg_official": float((i2t + t2i) / 2.0),
        "i2t_stable": float(records[0]["map_stable"]),
        "t2i_stable": float(records[1]["map_stable"]),
        "avg_stable": float((records[0]["map_stable"] + records[1]["map_stable"]) / 2.0),
        "n_query": int(len(query_labels)),
        "n_db": int(len(retrieval_labels)),
        "eval_records": records,
    }
