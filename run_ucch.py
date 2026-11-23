"""UCCH 官方代码的薄封装：只改运行环境，不改算法与超参逻辑。

为什么要封装：
  1. 上游 UCCH.py 在模块顶层（第 1 行）写死了 `seed = 123`，且在导入时即通过
     `from utils.config import args` 解析命令行参数 `sys.argv`。
  2. 上游 nets/__init__.py 导入了 Python 3.12 已移除的标准库 `imp`；
     上游 UCCH.py 顶层导入了 `torch.utils.tensorboard.SummaryWriter`（本地调试环境未装 tensorboard）。
  3. 上游 src/cmdataset.py 通过相对路径 `./data/...` 读取数据集。
  4. 上游评估函数 `fx_calc_map_multilabel_k` 使用 `dist.argsort()`（默认 quicksort，不稳定排序），
     在汉明距离存在大量并列（ties）时会有排序平局波动。

关于“导入时用 123 种子消耗的随机数是否影响结果”的核实说明：
  经逐行核查 `upstream/UCCH.py` 及其导入的全部子模块（`utils/config.py`、`nets/*`、`src/*`、`NCE/*`）：
  - `UCCH.py` 顶层（第 1~46 行）仅执行：设置 `seed = 123`、导入依赖库、解析 `sys.argv`、
    创建日志/检查点目录、定义 `main()` 与 `fx_calc_map_multilabel_k()` 函数；
  - 当通过正常 `import UCCH` 导入时，`__name__ == "UCCH" != "__main__"`，顶层**不会调用 `main()`**；
  - 数据集读取与打乱、`ImageNet`/`TextNet` 权重随机初始化、`NCEAverage` 记忆库随机初始化（`torch.randn`）
    以及 `DataLoader` 随机采样器（`shuffle=True`）**全部在 `main()` 内部构建和执行**。
  因此，模块导入期间在 `seed = 123` 下消耗的随机数次数严格为 **0**。
  导入完成后、调用 `UCCH.main()` 之前，按 `--seed` 重新执行与上游完全一致的种子设置序列
  （`UCCH.seed`、`np.random.seed`、`random.seed`、`PYTHONHASHSEED`、`torch.manual_seed`、
  `torch.cuda.manual_seed`、`torch.cuda.manual_seed_all`），即可保证进入 `main()` 时的随机数生成器状态
  与直接修改上游源码第 1 行 `seed = <指定值>` **逐位一致**；默认 `--seed 123` 时与上游完全逐位一致。

关于 `fx_calc_map_multilabel_k` 调用顺序的核实说明：
  在 `UCCH.main()` 中：
  - 训练阶段每个 epoch 结束调用一次 `test(epoch)`（默认 `is_eval=True`），其内部依次调用两次
    `fx_calc_map_multilabel_k`：第 1 次计算验证集 `i2t`（Img2Txt），第 2 次计算验证集 `t2i`（Txt2Img）；
  - 全部训练 epoch 结束后，加载验证集最佳模型权重，调用一次 `test(chp['epoch'], is_eval=False)`，
    其内部同样依次调用两次 `fx_calc_map_multilabel_k`：第 1 次计算测试集 `i2t`，第 2 次计算测试集 `t2i`。
  因此，总调用次数为 `2 * (max_epochs - start_epoch) + 2`：
  前 `2 * (max_epochs - start_epoch)` 次成对对应各 epoch 的验证集 `(i2t, t2i)`，
  **最后两次（`records[-2]` 与 `records[-1]`）严格对应最终测试集的 `i2t` 与 `t2i`**。

用法示例：
  python run_ucch.py --seed 0 --data_root /content/drive/MyDrive/UCCH_data --out results/mir_b128_s0 -- \
      --data_name mirflickr25k_fea --bit 128 --alpha 0.7 --num_hiden_layers 3 2 \
      --margin 0.2 --max_epochs 20 --train_batch_size 256 --shift 0.1 --lr 0.0001 --optimizer Adam
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import random
import shutil
import sys

# 禁止在 upstream/ 目录内生成 __pycache__ 字节码文件，确保 git -C upstream status --short 始终为空
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import scipy.spatial
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"


def ensure_compat_modules() -> List[str]:
    """注入缺失但实质无副作用的兼容替身模块（Python 3.12 的 imp、本地缺失的 tensorboard 等）。"""
    injected: List[str] = []

    try:
        importlib.import_module("imp")
    except ModuleNotFoundError:
        sys.modules["imp"] = types.ModuleType("imp")
        injected.append("imp")

    try:
        importlib.import_module("sympy")
    except ModuleNotFoundError:
        dummy_sympy = types.ModuleType("sympy")
        setattr(dummy_sympy, "arg", None)
        sys.modules["sympy"] = dummy_sympy
        injected.append("sympy")

    try:
        tb_loaded = importlib.import_module("torch.utils.tensorboard")
        getattr(tb_loaded, "SummaryWriter")
    except Exception:
        class _DummySummaryWriter:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def add_scalar(self, *args: Any, **kwargs: Any) -> None:
                pass

            def close(self) -> None:
                pass

        tb_mod = types.ModuleType("torch.utils.tensorboard")
        setattr(tb_mod, "SummaryWriter", _DummySummaryWriter)
        sys.modules["torch.utils.tensorboard"] = tb_mod
        injected.append("torch.utils.tensorboard")

    return injected


def patch_torch_load_for_pt26() -> bool:
    """PyTorch 2.6+ 默认 weights_only=True，上游 checkpoint 含 numpy 标量会报错，这里默认回退 False。"""
    orig_load = torch.load

    def _compat_load(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    torch.load = _compat_load  # type: ignore[assignment]
    return True


def link_data_directory(data_root: Path, dst_link: Path) -> str:
    """在运行目录创建指向 data_root 的 ./data 链接（兼容 Linux 符号链接与 Windows 目录联接）。"""
    if not data_root.exists():
        raise FileNotFoundError(f"指定的数据目录不存在: {data_root}")

    if dst_link.exists() or dst_link.is_symlink():
        try:
            if dst_link.resolve() == data_root.resolve():
                return "existing"
        except OSError:
            pass
        try:
            if dst_link.is_symlink() or os.path.islink(str(dst_link)):
                os.unlink(str(dst_link))
            else:
                os.rmdir(str(dst_link))
        except OSError as exc:
            raise RuntimeError(f"无法清理旧的 data 链接 {dst_link}: {exc}") from exc

    try:
        os.symlink(str(data_root), str(dst_link), target_is_directory=True)
        return "symlink"
    except OSError as symlink_exc:
        if os.name == "nt":
            try:
                import _winapi
                _winapi.CreateJunction(str(data_root), str(dst_link))
                return "junction"
            except Exception as junc_exc:
                raise RuntimeError(
                    f"Windows 下创建符号链接和目录联接均失败: symlink={symlink_exc}, junction={junc_exc}"
                ) from junc_exc
        raise RuntimeError(f"创建数据目录符号链接失败 ({dst_link} -> {data_root}): {symlink_exc}") from symlink_exc


def remove_data_link(dst_link: Path) -> None:
    """安全移除临时工作目录中的 ./data 链接，绝不删除真实数据文件。"""
    if not (dst_link.exists() or dst_link.is_symlink()):
        return
    try:
        if dst_link.is_symlink() or os.path.islink(str(dst_link)):
            os.unlink(str(dst_link))
        else:
            os.rmdir(str(dst_link))
    except OSError:
        pass


def parse_wrapper_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    if "--" in argv:
        cut = argv.index("--")
        own, passthrough = argv[:cut], argv[cut + 1:]
    else:
        p_known = argparse.ArgumentParser(add_help=False)
        p_known.add_argument("--seed", type=int, default=123)
        p_known.add_argument("--out", type=str, required=True)
        p_known.add_argument("--data_root", type=str, default="")
        p_known.add_argument("--deterministic", action="store_true")
        known, passthrough = p_known.parse_known_args(argv)
        return known, passthrough

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=123, help="随机种子（默认 123，与上游一致）")
    p.add_argument("--out", type=str, required=True, help="输出目录（存放 summary.json、logs、ckpt）")
    p.add_argument(
        "--data_root",
        type=str,
        default="",
        help="数据集根目录（下含 MIRFLICKR25K / NUS-WIDE-TC10 / MSCOCO 等子目录）",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="关闭 cudnn.benchmark 并开启确定性算法（默认关闭以保持与上游一致）",
    )
    return p.parse_args(own), passthrough


def resolve_data_root(arg_data_root: str) -> Path:
    if arg_data_root:
        return Path(arg_data_root).expanduser().resolve()
    env_root = os.environ.get("UCCH_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    colab_default = Path("/content/drive/MyDrive/UCCH_data")
    if colab_default.exists():
        return colab_default.resolve()
    return (UPSTREAM / "data").resolve()


def apply_seed(ucch_module: Any, seed: int, deterministic: bool) -> None:
    """按上游 UCCH.py 第 1~12 行完全相同的顺序重设随机种子。"""
    setattr(ucch_module, "seed", seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def compute_map_from_order(
    order_all: np.ndarray,
    retrieval_labels: np.ndarray,
    query_label: np.ndarray,
    k: int,
) -> float:
    """严格复现上游 fx_calc_map_multilabel_k 的 AP 计算逻辑。"""
    numcases = order_all.shape[0]
    res: List[float] = []
    for i in range(numcases):
        order = order_all[i].reshape(-1)[0:k]
        tmp_label = np.dot(retrieval_labels[order], query_label[i]) > 0
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res.append(float(np.dot(tmp_label, prec) / total_pos))
    if not res:
        return 0.0
    return float(np.mean(res))


def make_map_wrapper(
    records: List[Dict[str, Any]],
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str], float]:
    def map_both(
        retrieval: np.ndarray,
        retrieval_labels: np.ndarray,
        query: np.ndarray,
        query_label: np.ndarray,
        k: int = 0,
        metric: str = "cosine",
    ) -> float:
        # 显式使用字面量分支调用 cdist，满足 pyright 对 metric: _MetricKind 的严格类型检查
        if metric == "hamming":
            dist = scipy.spatial.distance.cdist(query, retrieval, metric="hamming")
        elif metric == "cosine":
            dist = scipy.spatial.distance.cdist(query, retrieval, metric="cosine")
        elif metric == "euclidean":
            dist = scipy.spatial.distance.cdist(query, retrieval, metric="euclidean")
        else:
            dist = scipy.spatial.distance.cdist(query, retrieval, metric="cosine")

        kk = int(dist.shape[1]) if k == 0 else int(k)
        # 官方实现：dist.argsort()（默认 quicksort/introsort，不稳定排序）
        ord_official = dist.argsort()
        map_official = compute_map_from_order(ord_official, retrieval_labels, query_label, kk)

        # 稳定排序对照：mergesort，保持距离并列样本在检索库中的原始先后顺序
        sort_mergesort: Literal["mergesort"] = "mergesort"
        ord_stable = dist.argsort(kind=sort_mergesort)
        map_stable = compute_map_from_order(ord_stable, retrieval_labels, query_label, kk)

        records.append(
            {
                "call_index": len(records),
                "metric": metric,
                "n_query": int(dist.shape[0]),
                "n_db": int(dist.shape[1]),
                "map_official": map_official,
                "map_stable": map_stable,
            }
        )
        return map_official

    return map_both


def build_epoch_records(eval_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将前 2 * N 条验证调用记录按每 epoch 的 (i2t, t2i) 成对整理。"""
    val_records = eval_records[:-2] if len(eval_records) >= 2 else []
    epoch_list: List[Dict[str, Any]] = []
    for idx in range(0, len(val_records) - 1, 2):
        rec_i2t = val_records[idx]
        rec_t2i = val_records[idx + 1]
        i2t_off = float(rec_i2t["map_official"])
        t2i_off = float(rec_t2i["map_official"])
        i2t_stb = float(rec_i2t["map_stable"])
        t2i_stb = float(rec_t2i["map_stable"])
        epoch_list.append(
            {
                "epoch": idx // 2,
                "val_i2t_official": i2t_off,
                "val_t2i_official": t2i_off,
                "val_avg_official": (i2t_off + t2i_off) / 2.0,
                "val_i2t_stable": i2t_stb,
                "val_t2i_stable": t2i_stb,
                "val_avg_stable": (i2t_stb + t2i_stb) / 2.0,
                "n_query": int(rec_i2t["n_query"]),
                "n_db": int(rec_i2t["n_db"]),
            }
        )
    return epoch_list


def main() -> None:
    wargs, passthrough = parse_wrapper_args(sys.argv[1:])
    out_dir = Path(wargs.out).expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.exit(f"创建输出目录失败 ({out_dir}): {exc}")

    data_root = resolve_data_root(wargs.data_root)
    injected_modules = ensure_compat_modules()
    patch_torch_load_for_pt26()

    orig_cwd = Path.cwd()
    work_dir_str = tempfile.mkdtemp(prefix="ucch_run_")
    work_dir = Path(work_dir_str).resolve()
    data_link = work_dir / "data"

    try:
        link_mode = link_data_directory(data_root, data_link)

        # 将 upstream 加入 sys.path 与 PYTHONPATH，确保主进程与 DataLoader 子进程均可导入上游模块
        upstream_str = str(UPSTREAM)
        if upstream_str not in sys.path:
            sys.path.insert(0, upstream_str)
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{upstream_str}{os.pathsep}{old_pythonpath}" if old_pythonpath else upstream_str
        )

        # 切换 cwd 到临时运行目录（内含 ./data -> data_root），不触碰 upstream 目录任何文件
        os.chdir(work_dir)
        sys.argv = ["UCCH.py", "--root_dir", str(out_dir), *passthrough]

        t0 = time.time()
        ucch = importlib.import_module("UCCH")

        # 修复 Windows / 非交互终端下 stty size 失败导致 term_width=10000 刷屏的问题（不影响随机数与数值）
        try:
            bar_show = importlib.import_module("utils.bar_show")
            if getattr(bar_show, "term_width", 80) > 200:
                setattr(bar_show, "term_width", 80)
        except Exception:
            pass

        # 重新按 --seed 执行与上游完全一致的种子初始化序列
        apply_seed(ucch, wargs.seed, wargs.deterministic)

        records: List[Dict[str, Any]] = []
        setattr(ucch, "fx_calc_map_multilabel_k", make_map_wrapper(records))

        # 执行上游训练与评估主流程
        ucch.main()
        wall_seconds = round(time.time() - t0, 2)

        if len(records) < 2:
            sys.exit(f"评估函数调用次数异常（共 {len(records)} 次，预期至少 2 次）")

        # 最后两次调用严格对应 test(chp['epoch'], is_eval=False) 中的测试集 i2t 与 t2i
        test_i2t_rec = records[-2]
        test_t2i_rec = records[-1]
        epoch_records = build_epoch_records(records)

        best_val_epoch = -1
        best_val_avg = -1.0
        for ep_rec in epoch_records:
            if ep_rec["val_avg_official"] > best_val_avg:
                best_val_avg = ep_rec["val_avg_official"]
                best_val_epoch = int(ep_rec["epoch"])

        parsed_args_dict: Dict[str, Any] = {}
        try:
            cfg_mod = importlib.import_module("utils.config")
            parsed_args_dict = dict(vars(getattr(cfg_mod, "args")))
        except Exception:
            pass

        test_i2t_off = float(test_i2t_rec["map_official"])
        test_t2i_off = float(test_t2i_rec["map_official"])
        test_i2t_stb = float(test_i2t_rec["map_stable"])
        test_t2i_stb = float(test_t2i_rec["map_stable"])

        summary: Dict[str, Any] = {
            "seed": int(wargs.seed),
            "deterministic": bool(wargs.deterministic),
            "data_root": str(data_root),
            "data_link_mode": link_mode,
            "injected_compat_modules": injected_modules,
            "upstream_args": passthrough,
            "parsed_args": parsed_args_dict,
            "best_val_epoch": best_val_epoch,
            "best_val_avg_official": best_val_avg if best_val_epoch >= 0 else None,
            "test_i2t_official": test_i2t_off,
            "test_t2i_official": test_t2i_off,
            "test_avg_official": (test_i2t_off + test_t2i_off) / 2.0,
            "test_i2t_stable": test_i2t_stb,
            "test_t2i_stable": test_t2i_stb,
            "test_avg_stable": (test_i2t_stb + test_t2i_stb) / 2.0,
            "n_query": int(test_i2t_rec["n_query"]),
            "n_db": int(test_i2t_rec["n_db"]),
            "epoch_records": epoch_records,
            "eval_records": records,
            "wall_seconds": wall_seconds,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

        summary_path = out_dir / "summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            sys.exit(f"写 summary.json 失败 ({summary_path}): {exc}")

        brief = {
            k: v
            for k, v in summary.items()
            if k not in ("eval_records", "epoch_records", "parsed_args")
        }
        print("SUMMARY", json.dumps(brief, ensure_ascii=False))
    finally:
        try:
            os.chdir(orig_cwd)
        except OSError:
            pass
        remove_data_link(data_link)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
