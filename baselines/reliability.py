"""外部方法的逐对可信度加权损失。"""
from __future__ import annotations
from typing import Any
import torch  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from baselines.methods import djsrh_loss, jdsh_loss, jdsh_similarity  # type: ignore[import-not-found]
from ours.nacg import djsrh_loss_nacg, jdsh_loss_nacg  # type: ignore[import-not-found]

__all__ = [
    "djsrh_loss_weighted",
    "jdsh_loss_weighted",
    "djsrh_loss_nacg",
    "jdsh_loss_nacg",
]


def djsrh_loss_weighted(image_features: torch.Tensor, texts: torch.Tensor, image_codes: torch.Tensor, text_codes: torch.Tensor, params: dict[str, Any], weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DJSRH 的三个相似度项按外积权重、元素数归一化。"""
    if bool(torch.all(weights == 1).item()):
        return djsrh_loss(image_features, texts, image_codes, text_codes, params)
    im, tx = F.normalize(image_features), F.normalize(texts)
    si, st = im.mm(im.t()) * 2.0 - 1.0, tx.mm(tx.t()) * 2.0 - 1.0
    sim = params["beta"] * si + (1.0 - params["beta"]) * st
    sim = ((1.0 - params["affinity_eta"]) * sim + params["affinity_eta"] * sim.mm(sim) / float(texts.shape[0])) * params["mu"]
    ib, tb = F.normalize(image_codes), F.normalize(text_codes)
    w = weights.detach().to(ib).view(-1)
    pair = w[:, None] * w[None, :]
    li = ((ib.mm(ib.t()) - sim).square() * pair).mean()
    lc = ((ib.mm(tb.t()) - sim).square() * pair).mean()
    lt = ((tb.mm(tb.t()) - sim).square() * pair).mean()
    return params["lambda1"] * li + lc + params["lambda2"] * lt, {"image": li, "cross": lc, "text": lt}


def jdsh_loss_weighted(image_features: torch.Tensor, texts: torch.Tensor, image_codes: torch.Tensor, text_codes: torch.Tensor, params: dict[str, Any], weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """JDSH 的四个相似度项和对角项分别加权。"""
    if bool(torch.all(weights == 1).item()):
        return jdsh_loss(image_features, texts, image_codes, text_codes, params)
    sim = jdsh_similarity(image_features, texts, params)
    ib, tb = F.normalize(image_codes, dim=1), F.normalize(text_codes, dim=1)
    w = weights.detach().to(ib).view(-1)
    pair = w[:, None] * w[None, :]
    def mse(x: torch.Tensor) -> torch.Tensor:
        return ((x - sim).square() * pair).mean()
    li = mse(ib.mm(ib.t()))
    lc = mse(ib.mm(tb.t())) + mse(tb.mm(ib.t())) - (w * (ib * tb).sum(dim=1)).mean()
    lt = mse(tb.mm(tb.t()))
    return params["intra"] * li + lc + params["intra"] * lt, {"image": li, "cross": lc, "text": lt, "similarity": sim}
