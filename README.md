# APSO Multimodal Biometric Authentication

Python implementation for the research work **A Dynamic Optimization Model for Multimodal Biometric Authentication**.

The repository contains an adaptive particle swarm optimization (APSO) approach to multimodal score-level biometric fusion, SVM authentication, held-out evaluation, and statistical utilities.

## Features

- Nine score-level fusion rules: sum, product, exponential, hyperbolic tangent, Einstein, Hamacher, Schweizer–Sklar, Frank, and Yager.
- APSO-inspired optimizer with fusion-rule selection and configurable decision threshold.
- Modality-specific SVM training and probability scoring.
- Saved touch and motion SVM example checkpoints and a portable prediction loader.
- FAR, FRR and context-weighted error computation.
- Held-out experiments and paired user-level bootstrap confidence intervals.
- Automated tests, numerical benchmark outputs, and an experimental PDF report.

## Project files

```text
APSO-Multimodal-Biometric-Authentication/
├── README.md
├── pyproject.toml
├── src/apso_auth/
│   ├── __init__.py
│   ├── fusion.py
│   ├── optimizer.py
│   ├── metrics.py
│   ├── svm.py
│   ├── pretrained.py
│   ├── evaluation.py
│   └── demo.py
├── experiments/
│   ├── benchmark.py
│   └── train_models.py
├── models/
│   ├── touch_svm.json
│   └── motion_svm.json
├── reports/
│   ├── APSO_Test_Report.pdf
│   └── benchmark_results.json
├── tests/
│   ├── test_core.py
│   └── test_svm.py
└── .github/workflows/tests.yml
```

## Installation

Python 3.10 or later:

```bash
git clone https://github.com/mohamedazab05/APSO-Multimodal-Biometric-Authentication.git
cd APSO-Multimodal-Biometric-Authentication
python -m pip install -e ".[dev]"
```

## Run tests and experiments

```bash
python -m pytest -q
apso-demo
python experiments/benchmark.py
```

The automated suite has passed five software tests locally.

- [Benchmark PDF report](reports/APSO_Test_Report.pdf)
- [Benchmark numeric results](reports/benchmark_results.json)
- [Benchmark Python script](experiments/benchmark.py)

**Benchmark data source:** The included example creates two-modality scores from a random generator. These measurements demonstrate the software workflow, not performance on the HMOG study dataset. The fixed-sum comparator is not a recreation of paper baselines [24] or [25].

## Pretrained SVM example models

The `models/` directory contains two **trained SVM example checkpoints** in JSON format:

| Model | Input feature count | Training source |
| --- | ---: | --- |
| `touch_svm.json` | 3 | Generated example features |
| `motion_svm.json` | 4 | Generated example features |

These models are genuinely fitted linear-kernel support vector classifiers with stored normalization, separating-hyperplane coefficients, and Platt calibration parameters. The portable loader approximates scikit-learn's calibrated probabilities; for exact full-library model serialization use the original trained `sklearn` pipeline and `joblib`. **Neither supplied checkpoint was trained on HMOG.**

Example inference:

```python
import numpy as np
from apso_auth.pretrained import predict_probability

touch_features = np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
touch_probabilities = predict_probability(touch_features, "models/touch_svm.json")
print(touch_probabilities)
```

To retrain the included example models using deterministic generated input features:

```bash
python experiments/train_models.py
```

For project-specific physiological or behavioral biometric features, retrain the per-modality SVMs on the actual study training set.

## Optimization objective

The context-weighted global error is

```text
E = CFA × FAR + (2 − CFA) × FRR
```

where `CFA` controls the relative cost of false acceptance and false rejection. The manuscript investigates 21 security levels from 0 to 2.

Example:

```python
import numpy as np
from apso_auth.evaluation import train_and_evaluate

learning = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.3], [0.7, 0.8]])
y_learning = np.array([0, 1, 0, 1])
evaluation = np.array([[0.15, 0.25], [0.85, 0.75], [0.3, 0.2], [0.9, 0.8]])
y_evaluation = np.array([0, 1, 0, 1])

model, held_out_error = train_and_evaluate(
    learning, y_learning, evaluation, y_evaluation,
    cfa=1.0, swarm_size=100, iterations=100, seed=42
)
print(model.rule, model.threshold, held_out_error)
```

## Dataset and experimental implementation

The original study uses the publicly available HMOG behavioral biometric dataset. The SVM interface in this project takes pre-extracted numerical touch and motion features. To reproduce the reported study, supply original feature extraction and partition settings, calibrate score distributions, use independent training/fusion/evaluation partitions, and rerun the claimed benchmarks on those real data.

The author has reported that a run of the implementation matched the experimental results. Input records, run logs, and the original HMOG evaluation outputs have not been supplied to this repository for independent comparison. The code is available and testable; exact reproduction of the paper's published numbers is a separate evaluation step.
