# Noise-Robust RUL Prediction via CET-MSA

This repository provides the replication package for the paper:

**Noise-Robust Remaining Useful Life Prediction for Intelligent Condition Monitoring via Convolutionally-Enhanced Temporal Self-Attention**
Please cite the paper when using this code:
Wang, Y., Zhang, J. & Qin, Y. Noise-robust remaining useful life prediction for intelligent condition monitoring via convolutionally-enhanced temporal self-attention. J Intell Manuf (2026). https://doi.org/10.1007/s10845-026-02895-3

## Overview

CET-MSA combines Temporal Local-Context Enhancement (TLCE) and Pyramidal Efficient Multi-Head Self-Attention (PE-MHSA) for noise-robust and efficient remaining useful life prediction. The implementation uses an LSTM temporal stem, TLCE key/value preconditioning, pyramidal key/value reduction, a compact regression head, and optional handcrafted degradation features.

## Baseline Attribution

The baseline code and experimental setting are derived from [ZhenghuaNTU/RUL-prediction-using-attention-based-deep-learning-approach](https://github.com/ZhenghuaNTU/RUL-prediction-using-attention-based-deep-learning-approach).
Please cite the original baseline paper when using this code:

Chen, Z., Wu, M., Zhao, R., Guretno, F., Yan, R., & Li, X. (2021). Machine remaining useful life prediction via an attention-based deep learning approach. IEEE Transactions on Industrial Electronics, 68(3), 2521-2531. https://doi.org/10.1109/TIE.2020.2972443

In this repository, the Chen et al. attention-based RUL model is treated as the baseline. The released implementation adds TLCE, PE-MHSA, CET-MSA, and degradation-aware transfer learning for noise-robust RUL prediction.

## Repository Structure

- `src/models/`: TLCE, PE-MHSA, and CET-MSA model code
- `src/data/`: C-MAPSS dataset loading, windowing, and handcrafted degradation features
- `src/training/`: training loop, checkpointing, and evaluation utilities
- `src/utils/`: metrics, seeding, and logging helpers
- `configs/`: experiment configurations for FD001-FD004 and transfer runs
- `scripts/`: preprocessing, training, evaluation, and benchmarking entry points
- `docs/`: environment, hardware, seeds, and reproduction details

## Data Preparation

The raw NASA C-MAPSS datasets are not included in this repository. Prepare the official files under:

```text
data/CMAPSS/
```

Expected files include `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, and the corresponding FD002-FD004 files.

Normalize a subset with training-set min/max statistics:

```bash
python scripts/preprocess_cmapss.py --subset FD001 --data-dir data/CMAPSS --processed-dir data/processed
```

Train CET-MSA on FD001:

```bash
python scripts/train.py --config configs/fd001.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --config configs/fd001.yaml --checkpoint checkpoints/FD001_iteration1_best_score.pth.tar
```

Run an inference benchmark:

```bash
python scripts/benchmark_inference.py --checkpoint checkpoints/FD001_iteration1_best_score.pth.tar
```
