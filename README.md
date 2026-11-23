# NCSG: Null-Calibrated Reliability and Signed Two-Hop Graphs for Unsupervised Cross-Modal Hashing under Noisy Correspondence

Official PyTorch implementation and verified experimental records for the manuscript:

> **Null-calibrated reliability and signed two-hop graphs for unsupervised cross-modal hashing under noisy correspondence**  
> Target journal: *Neurocomputing* (Elsevier)

---

## Overview

**NCSG** addresses unknown cross-modal pairing noise (*noisy correspondence*) in unsupervised cross-modal hashing (UCMH) without requiring prior knowledge of the noise rate, a clean validation subset, a second network, or extra inference-time cost:

1. **Off-Diagonal Null Calibration (ODNC):** Before training begins, 2-fold out-of-fold anchor-ridge agreement scores are calibrated against fixed-point-free derangement null scores constructed on the same corpus. On clean corpora ($\eta = 0$), the exact zero-noise gate returns $\hat{\eta}_0 = 0$ ($r_i \equiv 1$), avoiding clean-data degradation.
2. **Reliability-Gated Binary Memory & Sum-Normalized Contrast:** Pair reliabilities gate momentum binary memory writes ($r_i^2$ during warm-up, $r_i$ thereafter) and normalize contrastive and false-negative-masked ranking losses by the batch weight sum $W_B$.
3. **Sharpened Clean-Anchor Bridges & Signed Two-Hop Affinity Graph:** Row-centered signed two-hop profiles eliminate the positive inner-product bias of ReLU-truncated two-hop propagation, supplying accurate multi-label supervision.
4. **ODNC-Bridge Plug-In:** Cold-start reliability and clean-anchor bridges transfer directly to similarity-reconstruction hashing backbones (DJSRH, JDSH).

---

## Paper-to-CLI Alias Mapping

To ensure full traceability between archived experiment logs and paper terminology:

| Paper Name | CLI Flag / Variant | Description |
|---|---|---|
| **NCSG (ours)** | `ours/nacg_ucch_train.py --variant RCH-v6` | Full method ($\alpha=0.65, \lambda_{\text{graph}}=0.35, \beta_s=2.5$) |
| **NCSG ($\lambda_{\text{graph}}=0.22$)** | `ours/nacg_ucch_train.py --variant RCH-v6-con` | Conservative graph weight ($\alpha=0.70, \lambda_{\text{graph}}=0.22$) |
| **w/o graph & FN mask $(0,0)$** | `ours/nacg_ucch_train.py --variant RCH-v6-nog` | Reliability-gated backbone only |
| **FN mask only $(0,1)$** | `ours/nacg_ucch_train.py --variant RCH-v6-fnonly` | Single-factor FN ranking mask |
| **graph only $(1,0)$** | `ours/nacg_ucch_train.py --variant RCH-v6-gonly` | Single-factor signed two-hop graph loss |
| **graph + FN mask, static $(1,1)$** | `ours/nacg_ucch_train.py --variant RCH-v6-stat` | Static joint variant (no bridge refresh/sharpening) |
| **UCCH** | `ours/nacg_ucch_train.py --variant off` | Official UCCH protocol baseline |
| **UCCH+GMM** | `A-K1` (unconstrained symmetric 2-GMM baseline) | Unconstrained 2-component GMM gating baseline |
| **+ODNC-Bridge** | `baselines/run_external.py --method DJSRH-nacg / JDSH-nacg` | Plug-in transfer to DJSRH and JDSH |

---

## Quick Start & Self-Contained Verification Tests

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Standalone Unit & Anti-Leakage Audit Tests (No External Datasets Required)
```bash
pytest tests/ -v
```
This runs:
- `tests/test_no_leakage.py`: Static AST and runtime audits verifying zero access to labels, query splits, or dataset-specific branches inside `ours/nacg.py` and `ours/nacg_ucch_train.py`, and exact invariance (`0.0` diff) to ground-truth noise masks.
- `tests/test_ncsg_core.py`: Core unit tests covering ODNC exact zero-noise gating on clean data, noise separation at $\eta = 0.4$, signed two-hop graph differentiability, and ODNC-Bridge plug-in losses.

---

## Reproducing Full Benchmark Experiments

### 1. Fetch Official Third-Party Baselines
Third-party baseline repositories are **not redistributed** here to respect upstream licensing. Fetch them at the exact audited commits via:
```bash
bash scripts/fetch_upstreams.sh
```
This checks out:
- **UCCH**: `https://github.com/penghu-cs/UCCH` (`0c20e62b99875cd2ec9d7a496eae80b3ab8ba61b`) -> `upstream/`
- **DJSRH**: `https://github.com/zs-zhong/DJSRH` (`6041c2df810723dd0052e2e5b7c6bd33033f0f21`) -> `external/djsrh-upstream/`
- **JDSH**: `https://github.com/KaiserLew/JDSH` (`caa8224e2f2e028042fe270cab318c3b96590e41`) -> `external/jdsh-upstream/`

### 2. Prepare Public Benchmark Features
Download the standard pre-extracted feature files following the official [UCCH protocol](https://github.com/penghu-cs/UCCH) into `/path/to/datasets`:
- `MIRFLICKR25K/`: `mirflickr25k-iall-vgg.mat`, `mirflickr25k-yall.mat`, `mirflickr25k-lall.mat`
- `NUS-WIDE-TC10/`: `nus-wide-tc10-xall-vgg.mat`, `nus-wide-tc10-yall.mat`, `nus-wide-tc10-lall.mat`
- `MSCOCO/`: `MSCOCO_deep_doc2vec_data_rand.h5py`

### 3. Single-Run Training & Evaluation Commands

**NCSG (main method):**
```bash
python ours/nacg_ucch_train.py \
  --data_name mscoco_fea \
  --data_root /path/to/datasets \
  --variant RCH-v6 \
  --bit 64 \
  --noise_rate 0.4 \
  --noise_seed 0 \
  --seed 0 \
  --require_cuda \
  --out runs/mscoco_n0.4_b64_s0_ncsg
```
Replace `--data_name` with `mirflickr25k_fea` or `nus_wide_tc10_fea`, `--bit` with `16 | 32 | 64 | 128`, `--noise_rate` with `0.0 | 0.2 | 0.4 | 0.6`, and `--seed` with `0 | 1 | 2`.

**ODNC-Bridge plug-in on DJSRH / JDSH:**
```bash
python baselines/run_external.py \
  --method JDSH-nacg \
  --data_name nus_wide_tc10_fea \
  --data_root /path/to/datasets \
  --bit 64 \
  --noise_rate 0.4 \
  --noise_seed 0 \
  --seed 0 \
  --require_cuda \
  --out runs/nus_n0.4_b64_s0_jdsh_odnc_bridge
```

**Batch runners, summary validation, and PR / Top-K curve regeneration:**
- Full multi-bit, noise sweep, and plug-in grid (51 runs): `colab/run_nacg_final51.py`
- Orthogonal $2\times2$ single-factor ablation (18 runs): `colab/run_nacg_sfactor18.py`
- Strict summary integrity validator: `colab/summary_validator.py`
- Aggregation script: `colab/aggregate_nacg.py`
- Regenerating raw binary codes (`test_codes.npz`) and recomputing 21-point PR & Precision@$K$ curves:
  ```bash
  python colab/build_pr_curves.py \
    --mode generate_pr15 \
    --data_root /path/to/datasets \
    --out_root runs/pr15_repro \
    --keep_npz
  ```

---

## Verified Experimental Artifacts (`results/`)

The `results/` directory provides machine-readable audited aggregates matching all paper tables and figures:
- `results/nacg_final51_aggregate.json`: Multi-bit main results ($\eta=0.4$, 16–128 bits), clean-data ($\eta=0.0$), MIRFLICKR noise sweep ($\eta \in \{0.0, 0.2, 0.4, 0.6\}$), and plug-in transfer results (mean $\pm$ std across seeds `{0, 1, 2}`).
- `results/nacg_sfactor18_aggregate.json`: Orthogonal $2\times2$ single-factor ablation results isolating the signed two-hop graph loss (`+2.87 pp` on MS-COCO) and FN ranking mask (`+0.18 pp`).
- `results/pr_topk_curves_64bit_eta04.json`: Summary-verified seed-0 21-point PR and Precision@$K$ curves at 64 bits and $\eta=0.4$.

---

## License

Original source code in this repository is released under the [MIT License](LICENSE). Third-party datasets (MIRFLICKR-25K, NUS-WIDE, MS-COCO) and upstream baseline repositories (UCCH, DJSRH, JDSH) remain governed by their respective licenses and terms.
