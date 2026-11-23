from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler  # type: ignore[import-not-found]

BASELINE_DIR = Path(__file__).resolve().parent.parent
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from baselines.methods import (  # type: ignore[import-not-found]
    build_models,
    cpu_state_dict,
    djsrh_loss,
    jdsh_loss,
    method_defaults,
    restore_state_dict,
    source_registry,
)
from baselines.protocol import (  # type: ignore[import-not-found]
    apply_training_noise,
    build_data_manifest,
    evaluate_codes,
    load_protocol_data,
    save_noise_data,
    sha256_file,
    write_json_atomic,
)
from run_ucch import apply_seed  # type: ignore[import-not-found]
from ours.mcr import apply_feature_mask, compute_consistency_signal, compute_sample_weights  # type: ignore[import-not-found]
from ours.nacg import (  # type: ignore[import-not-found]
    NACGAnchorState,
    build_nacg_anchor_state,
    djsrh_loss_nacg,
    gather_bridged_features,
    jdsh_loss_nacg,
    update_nacg_epoch_reliability,
)
from ours.separation import fit_gmm_reliability_with_params, apply_separation_gate  # type: ignore[import-not-found]
from baselines.reliability import djsrh_loss_weighted, jdsh_loss_weighted  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]

OFFICIAL_TRAINING = {
    "DJSRH": {"num_epochs": 80, "batch_size": 32, "eval_interval": 40, "validation_metric": "mAP@50"},
    "JDSH": {"num_epochs": 300, "batch_size": 32, "eval_interval": 40, "validation_metric": "mAP@50"},
}


class FeatureDataset(Dataset):
    def __init__(self, images: np.ndarray, texts: np.ndarray, labels: np.ndarray) -> None:
        self.images = images
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        return self.images[index], self.texts[index], self.labels[index], index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DJSRH/JDSH官方核心算法的UCCH同协议适配")
    parser.add_argument("--method", required=True, choices=["DJSRH", "JDSH", "DJSRH-pi", "JDSH-pi", "DJSRH-nacg", "JDSH-nacg"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--data_name", required=True, choices=["mirflickr25k_fea", "nus_wide_tc10_fea", "mscoco_fea"])
    parser.add_argument("--bit", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--noise_rate", required=True, type=float)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data_manifest", default="")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--map_query_chunk_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_true")
    parser.add_argument("--dump_codes", action="store_true")
    parser.add_argument("--pi_gate", choices=["none", "ashman"], default="none")
    parser.add_argument("--pi_force_one", action="store_true")
    args = parser.parse_args()
    if args.bit <= 0 or args.max_epochs <= 0 or args.train_batch_size <= 0 or args.eval_batch_size <= 0 or args.map_query_chunk_size <= 0:
        parser.error("bit、epoch、batch size和MAP query chunk size必须为正整数")
    if not 0.0 <= args.noise_rate <= 1.0:
        parser.error("noise_rate必须位于[0,1]")
    return args


def encode(model: torch.nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(
                device=device, dtype=torch.float32, non_blocking=device.type == "cuda",
            )
            result = model(batch)
            code = result[-1] if isinstance(result, tuple) else result
            outputs.append(torch.sign(code).cpu().numpy().astype(np.int8))
    return np.concatenate(outputs, axis=0)


def encode_gpu(model: torch.nn.Module, values_gpu: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, int(values_gpu.shape[0]), batch_size):
            batch = values_gpu[start:start + batch_size]
            result = model(batch)
            code = result[-1] if isinstance(result, tuple) else result
            outputs.append(torch.sign(code).to(torch.int8))
    return torch.cat(outputs, dim=0).cpu().numpy()


def encode_both(
    image_model: torch.nn.Module,
    text_model: torch.nn.Module,
    images: np.ndarray,
    texts: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        encode(image_model, images, batch_size, device),
        encode(text_model, texts, batch_size, device),
    )


def validation_metrics(
    image_model: torch.nn.Module,
    text_model: torch.nn.Module,
    images: np.ndarray,
    texts: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    query_chunk_size: int,
) -> dict[str, Any]:
    size = min(2000, len(labels))
    images = images[:size]
    texts = texts[:size]
    labels = labels[:size]
    retrieval_images, retrieval_texts = encode_both(image_model, text_model, images, texts, batch_size, device)
    return evaluate_codes(
        retrieval_images,
        retrieval_texts,
        labels,
        retrieval_images,
        retrieval_texts,
        labels,
        query_chunk_size=query_chunk_size,
    )


def validation_metrics_gpu(
    image_model: torch.nn.Module,
    text_model: torch.nn.Module,
    val_images_gpu: torch.Tensor,
    val_texts_gpu: torch.Tensor,
    val_labels: np.ndarray,
    batch_size: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    retrieval_images = encode_gpu(image_model, val_images_gpu, batch_size)
    retrieval_texts = encode_gpu(text_model, val_texts_gpu, batch_size)
    return evaluate_codes(
        retrieval_images,
        retrieval_texts,
        val_labels,
        retrieval_images,
        retrieval_texts,
        val_labels,
        query_chunk_size=query_chunk_size,
    )


def build_optimizer(method: str, models: Any, params: dict[str, Any]) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    common = {"momentum": params["momentum"], "weight_decay": params["weight_decay"]}
    image = torch.optim.SGD(models.image.parameters(), lr=params["lr_img"], **common)
    text = torch.optim.SGD(models.text.parameters(), lr=params["lr_txt"], **common)
    return image, text


def encode_continuous(
    model: torch.nn.Module,
    values: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    is_tensor = isinstance(values, torch.Tensor)
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), batch_size):
            end = min(start + batch_size, int(values.shape[0]))
            batch = (
                values[start:end]
                if is_tensor
                else torch.from_numpy(values[start:end]).to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            )
            result = model(batch)
            code = result[-1] if isinstance(result, tuple) else result
            outputs.append(F.normalize(code, dim=1))
    return torch.cat(outputs, dim=0)


def train_epoch(
    method: str,
    models: Any,
    loader: DataLoader,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    params: dict[str, Any],
    device: torch.device,
    epoch: int,
    pi_buffer: torch.Tensor | None = None,
    nacg_state: NACGAnchorState | None = None,
    anc_norm_img: torch.Tensor | None = None,
    anc_norm_txt: torch.Tensor | None = None,
) -> float:
    models.image.train()
    models.text.train()
    if models.reference_image is not None:
        models.reference_image.eval()
    models.image.set_alpha(epoch)
    models.text.set_alpha(epoch)
    image_optimizer, text_optimizer = optimizers
    use_pi = pi_buffer is not None and not bool(torch.all(pi_buffer == 1).item())
    use_nacg = (
        nacg_state is not None
        and anc_norm_img is not None
        and anc_norm_txt is not None
        and not bool(torch.all(nacg_state.r == 1.0).item())
    )
    losses = []
    for images, texts, _, indices in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        texts = texts.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        image_optimizer.zero_grad()
        text_optimizer.zero_grad()
        image_features, _, image_codes = models.image(images)
        text_output = models.text(texts)
        text_codes = text_output[-1] if isinstance(text_output, tuple) else text_output
        if use_pi:
            assert pi_buffer is not None
            pi = pi_buffer[indices.to(pi_buffer.device)].to(device)
            weights = compute_sample_weights(pi, image_codes, text_codes, gamma=0.0, delta=0.1)
            if method == "DJSRH":
                loss, _ = djsrh_loss_weighted(images, texts, image_codes, text_codes, params, weights)
            else:
                loss, _ = jdsh_loss_weighted(image_features, texts, image_codes, text_codes, params, weights)
        elif use_nacg:
            assert nacg_state is not None and anc_norm_img is not None and anc_norm_txt is not None
            idx_n = indices.to(device=nacg_state.r.device, dtype=torch.long)
            r_b = nacg_state.r.index_select(0, idx_n).to(device)
            x_bridge = gather_bridged_features(
                anc_norm_img,
                nacg_state.nbr_idx_T.index_select(0, idx_n),
                nacg_state.nbr_w_T.index_select(0, idx_n),
            ).to(device)
            y_bridge = gather_bridged_features(
                anc_norm_txt,
                nacg_state.nbr_idx_I.index_select(0, idx_n),
                nacg_state.nbr_w_I.index_select(0, idx_n),
            ).to(device)
            if method == "DJSRH":
                loss, _ = djsrh_loss_nacg(images, texts, image_codes, text_codes, params, r_b, x_bridge, y_bridge)
            else:
                loss, _ = jdsh_loss_nacg(image_features, texts, image_codes, text_codes, params, r_b, x_bridge, y_bridge)
        elif method == "DJSRH":
            loss, _ = djsrh_loss(images, texts, image_codes, text_codes, params)
        else:
            loss, _ = jdsh_loss(image_features, texts, image_codes, text_codes, params)
        loss.backward()
        image_optimizer.step()
        text_optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses or not np.isfinite(losses).all():
        raise RuntimeError(f"{method} epoch {epoch}没有有效损失")
    return float(np.mean(losses))



def train_epoch_gpu(
    method: str,
    models: Any,
    gpu_images: torch.Tensor,
    gpu_texts: torch.Tensor,
    n_samples: int,
    train_batch_size: int,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    params: dict[str, Any],
    device: torch.device,
    epoch: int,
    pi_buffer_gpu: torch.Tensor | None = None,
    nacg_state: NACGAnchorState | None = None,
    anc_norm_img: torch.Tensor | None = None,
    anc_norm_txt: torch.Tensor | None = None,
) -> float:
    models.image.train()
    models.text.train()
    if models.reference_image is not None:
        models.reference_image.eval()
    models.image.set_alpha(epoch)
    models.text.set_alpha(epoch)
    image_optimizer, text_optimizer = optimizers
    use_pi = pi_buffer_gpu is not None and not bool(torch.all(pi_buffer_gpu == 1).item())
    use_nacg = (
        nacg_state is not None
        and anc_norm_img is not None
        and anc_norm_txt is not None
        and not bool(torch.all(nacg_state.r == 1.0).item())
    )
    # Consume the same single int64 draw that DataLoader._BaseDataLoaderIter.__init__ takes before RandomSampler.
    torch.empty((), dtype=torch.int64).random_()
    sampler = BatchSampler(RandomSampler(range(n_samples)), batch_size=train_batch_size, drop_last=True)
    losses: list[torch.Tensor] = []
    for batch_indices in sampler:
        idx = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
        images = gpu_images.index_select(0, idx)
        texts = gpu_texts.index_select(0, idx)
        image_optimizer.zero_grad()
        text_optimizer.zero_grad()
        image_features, _, image_codes = models.image(images)
        text_output = models.text(texts)
        text_codes = text_output[-1] if isinstance(text_output, tuple) else text_output
        if use_pi:
            assert pi_buffer_gpu is not None
            pi = pi_buffer_gpu.index_select(0, idx)
            weights = compute_sample_weights(pi, image_codes, text_codes, gamma=0.0, delta=0.1)
            if method == "DJSRH":
                loss, _ = djsrh_loss_weighted(images, texts, image_codes, text_codes, params, weights)
            else:
                loss, _ = jdsh_loss_weighted(image_features, texts, image_codes, text_codes, params, weights)
        elif use_nacg:
            assert nacg_state is not None and anc_norm_img is not None and anc_norm_txt is not None
            r_b = nacg_state.r.index_select(0, idx)
            x_bridge = gather_bridged_features(
                anc_norm_img,
                nacg_state.nbr_idx_T.index_select(0, idx),
                nacg_state.nbr_w_T.index_select(0, idx),
            )
            y_bridge = gather_bridged_features(
                anc_norm_txt,
                nacg_state.nbr_idx_I.index_select(0, idx),
                nacg_state.nbr_w_I.index_select(0, idx),
            )
            if method == "DJSRH":
                loss, _ = djsrh_loss_nacg(images, texts, image_codes, text_codes, params, r_b, x_bridge, y_bridge)
            else:
                loss, _ = jdsh_loss_nacg(image_features, texts, image_codes, text_codes, params, r_b, x_bridge, y_bridge)
        elif method == "DJSRH":
            loss, _ = djsrh_loss(images, texts, image_codes, text_codes, params)
        else:
            loss, _ = jdsh_loss(image_features, texts, image_codes, text_codes, params)
        loss.backward()
        image_optimizer.step()
        text_optimizer.step()
        losses.append(loss.detach())
    if not losses:
        raise RuntimeError(f"{method} epoch {epoch}没有有效损失")
    loss_values = torch.stack(losses).cpu().numpy()
    if not np.isfinite(loss_values).all():
        raise RuntimeError(f"{method} epoch {epoch}没有有效损失")
    return float(np.mean([float(v) for v in loss_values]))


def reliability_traversal(models: Any, loader: DataLoader, device: torch.device, generator: torch.Generator, seed: int, gate: str, noise_mask: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    """顺序遍历并以独立掩码流生成可信度。"""
    models.image.eval()
    models.text.eval()
    signals = []
    with torch.no_grad():
        for images, texts, _, _ in loader:
            images = images.to(device=device, dtype=torch.float32)
            texts = texts.to(device=device, dtype=torch.float32)
            def code(model: Any, value: torch.Tensor) -> torch.Tensor:
                output = model(value)
                return F.normalize(output[-1] if isinstance(output, tuple) else output, dim=1)
            hx, hy = code(models.image, images), code(models.text, texts)
            mx = apply_feature_mask(images, 0.8, generator)
            my = apply_feature_mask(texts, 0.8, generator)
            signals.append(compute_consistency_signal(hx, hy, code(models.image, mx), code(models.text, my)).cpu())
    result = fit_gmm_reliability_with_params(torch.cat(signals).numpy(), random_state=seed + 1000)
    pi, sep_s = apply_separation_gate(result) if gate == "ashman" else (result["pi"], None)
    clean = ~noise_mask
    auc = float(roc_auc_score(clean.astype(np.int32), pi)) if 0 < int(noise_mask.sum()) < len(noise_mask) else None
    diag = {"pi_mean": float(np.mean(pi)), "pi_quantiles": np.quantile(pi, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(), "pi_auc": auc, "pi_clean_mean": float(np.mean(pi[clean])) if clean.any() else None, "pi_noisy_mean": float(np.mean(pi[noise_mask])) if noise_mask.any() else None, "sep_D": result["D"] if gate == "ashman" else None, "sep_s": sep_s, "gmm_fallback": result["fallback"]}
    return torch.from_numpy(pi.copy()), diag


def reliability_traversal_gpu(
    models: Any,
    gpu_images: torch.Tensor,
    gpu_texts: torch.Tensor,
    eval_batch_size: int,
    generator: torch.Generator,
    seed: int,
    gate: str,
    noise_mask: np.ndarray,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """顺序遍历GPU常驻特征并以独立掩码流生成可信度。"""
    models.image.eval()
    models.text.eval()
    signals = []
    n_samples = int(gpu_images.shape[0])
    with torch.no_grad():
        for start in range(0, n_samples, eval_batch_size):
            end = min(start + eval_batch_size, n_samples)
            images = gpu_images[start:end]
            texts = gpu_texts[start:end]
            def code(model: Any, value: torch.Tensor) -> torch.Tensor:
                output = model(value)
                return F.normalize(output[-1] if isinstance(output, tuple) else output, dim=1)
            hx, hy = code(models.image, images), code(models.text, texts)
            mx = apply_feature_mask(images, 0.8, generator)
            my = apply_feature_mask(texts, 0.8, generator)
            signals.append(compute_consistency_signal(hx, hy, code(models.image, mx), code(models.text, my)))
    result = fit_gmm_reliability_with_params(torch.cat(signals, dim=0).cpu().numpy(), random_state=seed + 1000)
    pi, sep_s = apply_separation_gate(result) if gate == "ashman" else (result["pi"], None)
    clean = ~noise_mask
    auc = float(roc_auc_score(clean.astype(np.int32), pi)) if 0 < int(noise_mask.sum()) < len(noise_mask) else None
    diag = {"pi_mean": float(np.mean(pi)), "pi_quantiles": np.quantile(pi, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(), "pi_auc": auc, "pi_clean_mean": float(np.mean(pi[clean])) if clean.any() else None, "pi_noisy_mean": float(np.mean(pi[noise_mask])) if noise_mask.any() else None, "sep_D": result["D"] if gate == "ashman" else None, "sep_s": sep_s, "gmm_fallback": result["fallback"]}
    return torch.from_numpy(pi.copy()).to(device=gpu_images.device), diag


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("正式外部基线运行要求CUDA，但当前不可用")
    apply_seed(argparse.Namespace(), args.seed, args.deterministic)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_protocol_data(Path(args.data_root).expanduser().resolve(), args.data_name)
    noise = apply_training_noise(data, args.noise_rate, args.noise_seed)
    artifact_hashes = save_noise_data(out_dir, noise)
    manifest: dict[str, Any] | None = None
    manifest_sha256 = None
    if args.data_manifest:
        manifest_path = Path(args.data_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest is not None
        if manifest.get("data_name") != args.data_name:
            raise ValueError("数据清单与data_name不一致")
        manifest_sha256 = sha256_file(manifest_path)
        for item in manifest["files"]:
            current = Path(args.data_root) / item["path"]
            if current.stat().st_size != item["size_bytes"]:
                raise ValueError(f"数据文件大小与清单不一致: {current}")

    nacg_enabled = args.method.endswith("-nacg")
    pi_enabled = args.method.endswith("-pi")
    method_base = args.method.removesuffix("-pi").removesuffix("-nacg")
    models = build_models(method_base, args.bit, data.texts.shape[1])
    models.image.to(device)
    models.text.to(device)
    if models.reference_image is not None:
        models.reference_image.to(device)
        models.reference_image.requires_grad_(False)
    params = method_defaults(method_base, args.data_name)
    optimizers = build_optimizer(method_base, models, params)
    dataset = FeatureDataset(data.images, noise.train_texts, data.labels)
    if len(dataset) < args.train_batch_size:
        raise ValueError("训练样本不足一个完整batch")
    use_gpu_resident = device.type == "cuda" and os.environ.get("DISABLE_EXTERNAL_GPU_RESIDENT") != "1"
    loader = None if use_gpu_resident else DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    source = source_registry()["selected_methods"][method_base]
    adaptations = [
        "official_mini_batch_losses_and_official_TxtNet_retained",
        "raw_image_AlexNet_replaced_by_UCCH_preextracted_VGG_features",
        "MSCOCO_uses_predeclared_NUSWIDE_method_configuration_without_tuning",
        "training_text_permutation_only_retrieval_and_test_query_clean",
        "shared_UCCH_train_retrieval_query_slices_and_20_epochs_batch_256",
        "validation_every_epoch_on_first_2000_retrieval_pairs_quicksort_MAP_ALL",
        "best_validation_epoch_restored_before_test_query_is_encoded",
    ]
    config = {
        "run_key": f"{args.data_name}_n{args.noise_rate:g}_b{args.bit}_s{args.seed}_{args.method}",
        "method": args.method,
        "arguments": vars(args),
        "method_defaults": params,
        "official_training": OFFICIAL_TRAINING[method_base],
        "source": source,
        "protocol_adaptations": adaptations,
        "adapter_code_sha256": {
            name: sha256_file(BASELINE_DIR / name)
            for name in ("baselines/protocol.py", "baselines/methods.py", "baselines/run_external.py", "baselines/source_registry.json")
        },
        "data_manifest_sha256": manifest_sha256,
        "noise_artifact_sha256": artifact_hashes,
    }
    write_json_atomic(out_dir / "run_config.json", config)

    pi_buffer = torch.ones(len(dataset), dtype=torch.float32)
    mask_gen = torch.Generator(device=device.type).manual_seed(args.seed + 1000)
    loader_gen = torch.Generator().manual_seed(args.seed + 1001)
    pi_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, generator=loader_gen) if (pi_enabled and not use_gpu_resident) else None
    gpu_train_images: torch.Tensor | None = None
    gpu_train_texts: torch.Tensor | None = None
    gpu_val_images: torch.Tensor | None = None
    gpu_val_texts: torch.Tensor | None = None
    val_labels: np.ndarray | None = None
    pi_buffer_gpu: torch.Tensor | None = None
    if use_gpu_resident:
        gpu_train_images = torch.as_tensor(data.images, device=device, dtype=torch.float32).contiguous()
        gpu_train_texts = torch.as_tensor(noise.train_texts, device=device, dtype=torch.float32).contiguous()
        val_size = min(2000, len(data.labels))
        gpu_val_images = gpu_train_images[:val_size]
        gpu_val_texts = torch.as_tensor(data.texts[:val_size], device=device, dtype=torch.float32).contiguous()
        val_labels = data.labels[:val_size]
        pi_buffer_gpu = pi_buffer.to(device=device) if pi_enabled else None

    nacg_state: NACGAnchorState | None = None
    anc_norm_img: torch.Tensor | None = None
    anc_norm_txt: torch.Tensor | None = None
    if nacg_enabled:
        nacg_imgs = gpu_train_images if gpu_train_images is not None else torch.as_tensor(data.images, device=device, dtype=torch.float32)
        nacg_txts = gpu_train_texts if gpu_train_texts is not None else torch.as_tensor(noise.train_texts, device=device, dtype=torch.float32)
        nacg_state = build_nacg_anchor_state(
            images_gpu=nacg_imgs,
            texts_gpu=nacg_txts,
            seed=args.seed,
            noise_mask=noise.mask,
        )
        if args.pi_force_one:
            nacg_state.hat_eta_0 = 0.0
            nacg_state.r_0.fill_(1.0)
            nacg_state.r.fill_(1.0)
        anc_norm_img = nacg_state.norm_images.index_select(0, nacg_state.anchor_idx)
        anc_norm_txt = nacg_state.norm_texts.index_select(0, nacg_state.anchor_idx)

    best_epoch = -1
    best_score = float("-inf")
    best_states: dict[str, Any] | None = None
    epoch_records = []
    for epoch in range(args.max_epochs):
        epoch_started = time.perf_counter()
        if use_gpu_resident:
            assert gpu_train_images is not None and gpu_train_texts is not None and gpu_val_images is not None and gpu_val_texts is not None and val_labels is not None
            train_loss = train_epoch_gpu(
                method_base, models, gpu_train_images, gpu_train_texts,
                len(dataset), args.train_batch_size, optimizers, params, device, epoch,
                pi_buffer_gpu if pi_enabled else None,
                nacg_state=nacg_state if nacg_enabled else None,
                anc_norm_img=anc_norm_img if nacg_enabled else None,
                anc_norm_txt=anc_norm_txt if nacg_enabled else None,
            )
            validation = validation_metrics_gpu(
                models.image, models.text, gpu_val_images, gpu_val_texts, val_labels,
                args.eval_batch_size, args.map_query_chunk_size,
            )
        else:
            assert loader is not None
            train_loss = train_epoch(
                method_base, models, loader, optimizers, params, device, epoch,
                pi_buffer if pi_enabled else None,
                nacg_state=nacg_state if nacg_enabled else None,
                anc_norm_img=anc_norm_img if nacg_enabled else None,
                anc_norm_txt=anc_norm_txt if nacg_enabled else None,
            )
            validation = validation_metrics(
                models.image, models.text, data.images, data.texts, data.labels,
                args.eval_batch_size, device, args.map_query_chunk_size,
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "val_i2t_official": validation["i2t_official"],
            "val_t2i_official": validation["t2i_official"],
            "val_avg_official": validation["avg_official"],
            "val_i2t_stable": validation["i2t_stable"],
            "val_t2i_stable": validation["t2i_stable"],
            "val_avg_stable": validation["avg_stable"],
            "n_query": validation["n_query"],
            "n_db": validation["n_db"],
        }
        if pi_enabled:
            if use_gpu_resident:
                assert gpu_train_images is not None and gpu_train_texts is not None
                pi_buffer_gpu, pi_diag = reliability_traversal_gpu(
                    models, gpu_train_images, gpu_train_texts,
                    args.eval_batch_size, mask_gen, args.seed, args.pi_gate, noise.mask,
                )
                if args.pi_force_one:
                    pi_buffer_gpu.fill_(1.0)
            else:
                assert pi_loader is not None
                pi_buffer, pi_diag = reliability_traversal(models, pi_loader, device, mask_gen, args.seed, args.pi_gate, noise.mask)
                if args.pi_force_one:
                    pi_buffer.fill_(1.0)
            record["pi_diagnostics"] = pi_diag
            record["epoch_seconds"] = time.perf_counter() - epoch_started
        elif nacg_enabled and nacg_state is not None:
            if args.pi_force_one or nacg_state.hat_eta_0 <= 0.0:
                empty_t = torch.empty((0, args.bit), device=device)
                _, nacg_diag = update_nacg_epoch_reliability(
                    nacg_state,
                    empty_t,
                    empty_t,
                    epoch=epoch,
                    noise_mask=noise.mask,
                    force_one=args.pi_force_one,
                )
            else:
                img_src = gpu_train_images if gpu_train_images is not None else data.images
                txt_src = gpu_train_texts if gpu_train_texts is not None else noise.train_texts
                hx_all = encode_continuous(models.image, img_src, args.eval_batch_size, device)
                hy_all = encode_continuous(models.text, txt_src, args.eval_batch_size, device)
                _, nacg_diag = update_nacg_epoch_reliability(
                    nacg_state,
                    hx_all,
                    hy_all,
                    epoch=epoch,
                    noise_mask=noise.mask,
                    force_one=False,
                )
            record["pi"] = nacg_diag
            record["pi_diagnostics"] = nacg_diag
            record["epoch_seconds"] = time.perf_counter() - epoch_started
        epoch_records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation["avg_official"] > best_score:
            best_score = validation["avg_official"]
            best_epoch = epoch
            best_states = {"image": cpu_state_dict(models.image), "text": cpu_state_dict(models.text)}
            if not args.no_save_ckpt:
                torch.save({"epoch": epoch, **best_states}, out_dir / "best_checkpoint.pt")
    if best_states is None:
        raise RuntimeError("没有可用的验证集最佳模型")
    restore_state_dict(models.image, best_states["image"])
    restore_state_dict(models.text, best_states["text"])

    if use_gpu_resident:
        assert gpu_train_images is not None
        del gpu_train_texts, gpu_val_images, gpu_val_texts
        if pi_buffer_gpu is not None:
            del pi_buffer_gpu
        retrieval_images = encode_gpu(models.image, gpu_train_images, args.eval_batch_size)
        del gpu_train_images
        gpu_clean_texts = torch.as_tensor(data.texts, device=device, dtype=torch.float32).contiguous()
        retrieval_texts = encode_gpu(models.text, gpu_clean_texts, args.eval_batch_size)
        del gpu_clean_texts
        query_images, query_texts = encode_both(models.image, models.text, data.query_images, data.query_texts, args.eval_batch_size, device)
    else:
        retrieval_images, retrieval_texts = encode_both(models.image, models.text, data.images, data.texts, args.eval_batch_size, device)
        query_images, query_texts = encode_both(models.image, models.text, data.query_images, data.query_texts, args.eval_batch_size, device)
    test = evaluate_codes(
        retrieval_images,
        retrieval_texts,
        data.labels,
        query_images,
        query_texts,
        data.query_labels,
        query_chunk_size=args.map_query_chunk_size,
    )
    if args.dump_codes:
        np.savez_compressed(
            out_dir / "test_codes.npz",
            retrieval_images=retrieval_images,
            retrieval_texts=retrieval_texts,
            retrieval_labels=data.labels,
            query_images=query_images,
            query_texts=query_texts,
            query_labels=data.query_labels,
        )
    trainable = {
        "image": sum(parameter.numel() for parameter in models.image.parameters() if parameter.requires_grad),
        "text": sum(parameter.numel() for parameter in models.text.parameters() if parameter.requires_grad),
    }
    summary = {
        "seed": args.seed,
        "deterministic": args.deterministic,
        "variant": args.method,
        "run_key": config["run_key"],
        "noise_rate": args.noise_rate,
        "noise_seed": args.noise_seed,
        "n_noisy": int(noise.mask.sum()),
        "n_train": data.n_train,
        "label_inconsistency": noise.label_diagnostic,
        "external_hyperparams": {
            **params,
            "max_epochs": args.max_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "map_query_chunk_size": args.map_query_chunk_size,
            "source": source,
            "official_training": OFFICIAL_TRAINING[method_base],
            "protocol_adaptations": adaptations,
        },
        "data_manifest": manifest,
        "data_manifest_sha256": manifest_sha256,
        "run_config_sha256": sha256_file(out_dir / "run_config.json"),
        "noise_artifact_sha256": artifact_hashes,
        "parsed_args": {
            "data_name": args.data_name,
            "bit": args.bit,
            "max_epochs": args.max_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "map_query_chunk_size": args.map_query_chunk_size,
            "num_workers": args.num_workers,
        },
        "best_val_epoch": best_epoch,
        "best_val_avg_official": best_score,
        "test_i2t_official": test["i2t_official"],
        "test_t2i_official": test["t2i_official"],
        "test_avg_official": test["avg_official"],
        "test_i2t_stable": test["i2t_stable"],
        "test_t2i_stable": test["t2i_stable"],
        "test_avg_stable": test["avg_stable"],
        "n_query": test["n_query"],
        "n_db": test["n_db"],
        "epoch_records": epoch_records,
        "eval_records": test["eval_records"],
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "trainable_parameters": trainable,
        "trainable_parameters_total": sum(trainable.values()),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": device.type == "cuda",
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    if pi_enabled:
        summary.update({"method_base": method_base, "pi_gate": args.pi_gate, "delta": 0.1, "pi_force_one": args.pi_force_one})
    if nacg_enabled and nacg_state is not None:
        summary.update(
            {
                "method_base": method_base,
                "pi_force_one": args.pi_force_one,
                "nacg_diagnostics": {
                    "hat_eta_0": float(nacg_state.hat_eta_0),
                    "hat_eta_raw_0": float(nacg_state.hat_eta_raw_0),
                    "cold_start_auc": nacg_state.cold_start_auc,
                    "zero_noise_exact_gate": bool(nacg_state.hat_eta_0 <= 0.0),
                    "last_epoch_pi": epoch_records[-1].get("pi") if epoch_records else None,
                },
            }
        )
    write_json_atomic(out_dir / "summary.json", summary)
    print("SUMMARY " + json.dumps({key: value for key, value in summary.items() if key not in {"epoch_records", "eval_records", "data_manifest"}}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
