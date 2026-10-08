# APSO Multimodal Biometric Authentication

**Research implementation accompanying** *A Dynamic Optimization Model for Multimodal Biometric Authentication*.

This Python package provides a modular implementation of an adaptive-particle-swarm-optimization (APSO) approach to multimodal biometric **score-level fusion**. It supports separate unimodal SVM score generation, fusion strategy selection, context-dependent error costs, and paired user-level bootstrap analysis.

> **Reproducibility status (October 2026):** The research author reports obtaining results matching their experiments using the sample implementation. This report is encouraging, but the specific input data, evaluation protocol, numerical comparisons, and run artifacts have **not yet been independently inspected or verified in this repository**. The included demo generates synthetic data and is **not evidence of reproducing the paper's published tables or figures**.

## Features

| Module | Capabilities |
| --- | --- |
| Fusion | Nine families: weighted sum, weighted product, exponential, hyperbolic tangent, Einstein, Hamacher, Schweizer–Sklar, Frank and Yager |
| Optimizer | APSO-inspired swarm optimization, diversity adaptation, elitist perturbation, fusion-rule selection and threshold selection |
| Classifiers | SVM-based modality-specific authentication from pre-extracted, aligned feature arrays |
| Metrics | FAR, FRR and context-weighted global error |
| Evaluation | Separate fusion training and held-out evaluation |
| Statistics | Paired, user-cluster bootstrap intervals for differences between systems |
| Verification | Automated tests and a small synthetic demonstration |

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── .gitignore
├── data/
│   └── .gitkeep
├── src/
│   └── apso_auth/
│       ├── __init__.py       # package interface
│       ├── fusion.py         # nine score-fusion functions
│       ├── optimizer.py      # APSO-inspired optimization
│       ├── metrics.py        # FAR, FRR, weighted error
│       ├── svm.py            # modality SVM training/scoring
│       ├── evaluation.py     # holdout evaluation / user bootstrap
│       └── demo.py           # synthetic example
└── tests/
    ├── test_core.py         # fusion, optimizer and bootstrap tests
    └── test_svm.py          # SVM scoring test
```

**All Python source files and tests shown above are committed in the repository.**

## Installation

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/mohamedazab05/APSO-Multimodal-Biometric-Authentication.git
cd APSO-Multimodal-Biometric-Authentication
python -m pip install -e ".[dev]"
```

Run the automated tests:

```bash
python -m pytest -q
```

Run the **synthetic** demonstration:

```bash
apso-demo
```

The demo prints selected fusion rules, decision thresholds and held-out costs for three example security coefficients. **It does not use the research dataset.**

## Quick example

```python
import numpy as np
from apso_auth.evaluation import train_and_evaluate

# Example only: scores from TWO pre-trained, calibrated modality models.
# Rows: evaluation attempts; columns: modalities.
learning_scores = np.array([
    [0.1, 0.2], [0.8, 0.9], [0.2, 0.3],
    [0.7, 0.8], [0.3, 0.2], [0.9, 0.7]
])
learning_labels = np.array([0, 1, 0, 1, 0, 1])

heldout_scores = np.array([
    [0.2, 0.1], [0.7, 0.9], [0.3, 0.4], [0.9, 0.8]
])
heldout_labels = np.array([0, 1, 0, 1])

result, heldout_error = train_and_evaluate(
    learning_scores, learning_labels,
    heldout_scores, heldout_labels,
    cfa=1.0, swarm_size=20, iterations=15, seed=42
)
print(result.rule, result.threshold, heldout_error)
```

For the reported manuscript configuration, specify `swarm_size=100` and `iterations=100`. Results will still depend on the actual feature extraction, data split, parameterization, input scores and random seed.

## Objective and security levels

The optimization minimizes the cost-weighted global error:

```text
E = CFA × FAR + (2 − CFA) × FRR
```

where `FAR` denotes false acceptance rate, `FRR` false rejection rate, and `CFA` is a scenario-dependent cost coefficient between **0 and 2**. The paper evaluates **21 values from 0 to 2 in increments of 0.1**. The context-to-`CFA` mapping is outside the current implementation.

## Reproducing research experiments

A credible replication requires more than running the demo:

1. Obtain the original **HMOG** behavioral biometric dataset from its authorized public source. Do not commit raw subject recordings to this repository.
2. Implement and document the original touch/motion feature extraction and alignment; the current SVM interface expects **already extracted numerical features**.
3. Use disjoint unimodal training, fusion-learning and fusion-evaluation partitions. Verify exact enrollment, session and user-level separation against the actual experimental protocol.
4. Record the original APSO parameter bounds and update equations, baseline implementations, calibration procedure, random seeds and thresholds.
5. Evaluate all security levels on held-out data; retain scores/labels/user IDs and save machine-readable outputs for audit.
6. Compute paired user-level bootstrap confidence intervals on the **actual held-out results** for the same evaluation population and fixed fitted fusion strategies.
7. Compare resulting plots, numerical tables and timings against the manuscript before labeling a run a verified reproduction.

### Current implementation differences and limitations

- This repository implements an **independent APSO-inspired variant** with diversity adaptation and elitist perturbation. It has not been demonstrated to be algorithmically identical to the authors' original APSO.
- Fusion parameter ranges and some nonlinear rule forms are implementation choices requiring comparison with the original equations/code.
- Unimodal models operate on **pre-extracted features**; HMOG preprocessing and feature engineering are not implemented here.
- The synthetic demo is not an evaluation of HMOG or published experimental outcomes.
- The reported bootstrap example values discussed during manuscript preparation are **not represented as measurements** in this project.

## Data, code and model availability

**Code:** All current Python source files are publicly available in this repository.

**Dataset:** The underlying behavioral biometric data are not hosted here. Refer to the dataset citation and official distribution instructions in the manuscript.

**Models:** No pretrained study SVM models are currently included.

**License:** No open-source license has yet been selected by the repository owner. Public visibility alone does not grant permission to reuse or redistribute the source.

## Citation

If you use this repository in research, cite the associated paper using its final published bibliographic information. A DOI and publication record should be added here when verified.

## Research integrity

Matching numerical outputs on a reported run is valuable evidence, but **verification requires the data partition, scripts, implementation equivalence, outputs and evaluation protocol to be archived and examined**. Please do not cite synthetic demonstrations or unverified example statistics as empirical findings.
