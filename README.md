# A Dynamic Optimization Model for Multimodal Biometric Authentication

**Independent research prototype — NOT a verified reproduction of the published experiments.**

This repository implements a modular Python toolkit inspired by the paper: nine score-level fusion rules, context-weighted FAR/FRR, an adaptive PSO optimizer with an elitist perturbation heuristic, modality-specific SVMs, separate fusion learning/evaluation, and a paired user-cluster bootstrap.

## Installation

```bash
python -m pip install -e '.[dev]'
pytest -q
apso-demo
```

## Source structure

- `src/apso_auth/fusion.py`: weighted sum/product/exp/tanh, Einstein, Hamacher, Schweizer–Sklar, Frank and Yager fusion.
- `src/apso_auth/optimizer.py`: independent APSO-inspired heuristic and per-rule optimization.
- `src/apso_auth/metrics.py`: FAR, FRR, global cost `CFA*FAR + (2-CFA)*FRR`.
- `src/apso_auth/svm.py`: SVM probability scores from pre-extracted, aligned modality features.
- `src/apso_auth/evaluation.py`: held-out fusion evaluation and paired user-cluster confidence intervals.
- `src/apso_auth/demo.py`: synthetic-only demonstration.
- `tests/`: automated checks.

## Scientific limitations and reproducibility

The manuscript specifies nine fusion rules, CFA in [0,2] in increments of 0.1, swarm size 100, 100 iterations, and SVM-based authenticators. It does **not** fully specify the original APSO state-estimation, parameter limits, ELS scale, score calibration, HMOG feature extraction, or partitioning details. This implementation makes independent documented choices and is **not** the authors' original code. Its sample results are synthetic and must not be used to support the manuscript's figures, tables, statistical significance claims, or measured runtimes.

**Dataset:** The publicly available behavioral biometric dataset is cited as Q. Yang et al. (2014), manuscript reference [32]. Obtain it from the approved source; no user recordings or biometric data are redistributed here.

Before claiming experimental reproduction, validate that the 70:30 SVM split and 1:1 fusion split are genuinely disjoint; preserve user/session identity; obtain the original feature extraction, scores, thresholds and experimental seeds; rerun the PSO baselines; and calculate significance from actual held-out evaluation results, not illustrative values.

No license is included until selected by the repository owner.
