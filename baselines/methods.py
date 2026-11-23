from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = Path(__file__).resolve().parent / "source_registry.json"


@dataclass
class MethodModels:
    image: nn.Module
    text: nn.Module
    reference_image: nn.Module | None


class DJSRHFeatureImgNet(nn.Module):
    def __init__(self, code_len: int) -> None:
        super().__init__()
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = x.view(x.size(0), -1)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch: int) -> None:
        self.alpha = (float(epoch) + 1.0) ** 0.5


class JDSHFeatureImgNet(nn.Module):
    def __init__(self, code_len: int) -> None:
        super().__init__()
        self.hash_layer = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            feat = F.normalize(x.view(x.size(0), -1), dim=1)
        hid = self.hash_layer(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch: int) -> None:
        self.alpha = (float(epoch) + 1.0) ** 0.5


def source_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def load_official_models_module(method: str) -> ModuleType:
    source = source_registry()["selected_methods"][method]
    upstream_dir = BASELINE_DIR / "external" / source["local_dir"]
    path = upstream_dir / source["models_file"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != source["models_sha256"]:
        raise RuntimeError(f"{method}官方models.py哈希不匹配: {actual}")
    train_path = upstream_dir / ("train.py" if method == "DJSRH" else "JDSH.py")
    train_actual = hashlib.sha256(train_path.read_bytes()).hexdigest()
    if train_actual != source["train_sha256"]:
        raise RuntimeError(f"{method}官方训练源码哈希不匹配: {train_actual}")
    spec = importlib.util.spec_from_file_location(f"_official_{method.lower()}_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_models(method: str, bit: int, text_dim: int) -> MethodModels:
    official = load_official_models_module(method)
    text = official.TxtNet(code_len=bit, txt_feat_len=text_dim)
    if method == "DJSRH":
        return MethodModels(DJSRHFeatureImgNet(bit), text, DJSRHFeatureImgNet(bit))
    if method == "JDSH":
        return MethodModels(JDSHFeatureImgNet(bit), text, None)
    raise ValueError(f"未知外部方法: {method}")


def method_defaults(method: str, data_name: str) -> dict[str, Any]:
    if method == "DJSRH":
        return {
            "beta": 0.9 if data_name == "mirflickr25k_fea" else 0.6,
            "lambda1": 0.1,
            "lambda2": 0.1,
            "mu": 1.5,
            "affinity_eta": 0.4,
            "lr_img": 0.001,
            "lr_txt": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "coco_config_source": "NUSWIDE",
        }
    if method == "JDSH":
        mir = data_name == "mirflickr25k_fea"
        return {
            "alpha": 0.5 if mir else 0.4,
            "beta": 0.1 if mir else 0.3,
            "lamb": 0.4 if mir else 0.3,
            "mu": 1.4,
            "intra": 0.1,
            "min": -0.52 if mir else -0.64,
            "max": 0.68 if mir else 0.75,
            "alpha_left": 2.0,
            "loc_left": -0.49 if mir else -0.62,
            "scale_left": 0.019 if mir else 0.0128,
            "beta_right": 4.0,
            "loc_right": -0.49 if mir else -0.62,
            "scale_right": 0.088 if mir else 0.086,
            "l1": 0.1 if mir else 0.2,
            "l2": 0.2,
            "lr_img": 0.01,
            "lr_txt": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "coco_config_source": "NUSWIDE",
        }
    raise ValueError(f"未知外部方法: {method}")


def djsrh_loss(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    image_codes: torch.Tensor,
    text_codes: torch.Tensor,
    params: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image_norm = F.normalize(image_features)
    text_norm = F.normalize(texts)
    similarity_image = image_norm.mm(image_norm.t()) * 2.0 - 1.0
    similarity_text = text_norm.mm(text_norm.t()) * 2.0 - 1.0
    similarity = params["beta"] * similarity_image + (1.0 - params["beta"]) * similarity_text
    similarity = (1.0 - params["affinity_eta"]) * similarity + params["affinity_eta"] * similarity.mm(similarity) / float(texts.shape[0])
    similarity = similarity * params["mu"]
    image_binary = F.normalize(image_codes)
    text_binary = F.normalize(text_codes)
    loss_image = F.mse_loss(image_binary.mm(image_binary.t()), similarity)
    loss_cross = F.mse_loss(image_binary.mm(text_binary.t()), similarity)
    loss_text = F.mse_loss(text_binary.mm(text_binary.t()), similarity)
    total = params["lambda1"] * loss_image + loss_cross + params["lambda2"] * loss_text
    return total, {"image": loss_image, "cross": loss_cross, "text": loss_text}


def jdsh_similarity(image_features: torch.Tensor, texts: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    image_norm = F.normalize(image_features)
    text_norm = F.normalize(texts)
    similarity_image = image_norm.mm(image_norm.t()) * 2.0 - 1.0
    similarity_text = text_norm.mm(text_norm.t()) * 2.0 - 1.0
    high = F.normalize(similarity_image).mm(F.normalize(similarity_text).t())
    similarity = params["alpha"] * similarity_image + params["beta"] * similarity_text + params["lamb"] * (high + high.t()) / 2.0
    left = params["loc_left"] - params["alpha_left"] * params["scale_left"]
    right = params["loc_right"] + params["beta_right"] * params["scale_right"]
    low = similarity < left
    high_mask = similarity > right
    similarity = similarity.clone()
    similarity[low] = (1.0 + params["l1"] * torch.exp(-(similarity[low] - params["min"]))) * similarity[low]
    similarity[high_mask] = (1.0 + params["l2"] * torch.exp(similarity[high_mask] - params["max"])) * similarity[high_mask]
    return similarity * params["mu"]


def jdsh_loss(
    image_features: torch.Tensor,
    texts: torch.Tensor,
    image_codes: torch.Tensor,
    text_codes: torch.Tensor,
    params: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    similarity = jdsh_similarity(image_features, texts, params)
    image_binary = F.normalize(image_codes, dim=1)
    text_binary = F.normalize(text_codes, dim=1)
    image_image = image_binary.mm(image_binary.t())
    text_text = text_binary.mm(text_binary.t())
    image_text = image_binary.mm(text_binary.t())
    text_image = text_binary.mm(image_binary.t())
    loss_image = F.mse_loss(image_image, similarity)
    loss_cross = F.mse_loss(image_text, similarity) + F.mse_loss(text_image, similarity) - (image_binary * text_binary).sum(dim=1).mean()
    loss_text = F.mse_loss(text_text, similarity)
    total = params["intra"] * loss_image + loss_cross + params["intra"] * loss_text
    return total, {"image": loss_image, "cross": loss_cross, "text": loss_text, "similarity": similarity}


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(copy.deepcopy(state))
