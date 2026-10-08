"""Nine score-level fusion families in the manuscript.

Inputs must be calibrated similarity/confidence scores in [0, 1].
For the nonlinear t-norms, implementation uses conventional, mathematically
valid parameterizations. Exact parameter constraints and formulas in the paper
should be checked against the original code before claims of replication.
"""
from __future__ import annotations
import numpy as np

RULES = ("sum", "product", "exp", "tanh", "einstein", "hamacher",
         "schweizer", "frank", "yager")
EPS = 1e-12


def _pair_tnorm(a: np.ndarray, b: np.ndarray, rule: str, p: float = 1.0) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    if rule == "einstein":
        return a * b / np.maximum(2 - a - b + a * b, EPS)
    if rule == "hamacher":
        return np.divide(a * b, a + b - a * b,
                         out=np.zeros_like(a), where=(a + b - a * b) > EPS)
    if rule == "schweizer":
        p = max(float(p), 1e-4)
        aa = np.maximum(a, EPS)
        bb = np.maximum(b, EPS)
        out = (aa**(-p) + bb**(-p) - 1)**(-1/p)
        return np.where((a == 0) | (b == 0), 0., out)
    if rule == "frank":
        p = max(float(p), 1e-4)
        if abs(p - 1) < 1e-5:
            return a * b
        return np.log1p(np.expm1(np.log(p)*a)*np.expm1(np.log(p)*b)/np.expm1(np.log(p))) / np.log(p)
    if rule == "yager":
        p = max(float(p), 1e-4)
        return np.maximum(1 - ((1-a)**p + (1-b)**p)**(1/p), 0)
    raise ValueError(f"Unknown t-norm: {rule}")


def fuse(scores: np.ndarray, rule: str, parameters: np.ndarray | None = None) -> np.ndarray:
    """Fuse shape (n_samples, n_modalities) -> (n_samples,)."""
    x = np.asarray(scores, float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.all(np.isfinite(x)) or np.any((x<0)|(x>1)):
        raise ValueError("Scores must have shape (samples, modalities>=2) and values in [0,1]")
    if rule not in RULES:
        raise ValueError(f"Unknown fusion rule {rule}")
    if rule in RULES[:4]:
        w = np.ones(x.shape[1])/x.shape[1] if parameters is None else np.asarray(parameters, float)
        if w.shape != (x.shape[1],) or np.any(w<0) or np.sum(w)<=0:
            raise ValueError("Invalid nonnegative modality weights")
        w = w / w.sum()
        if rule == "sum": return x @ w
        if rule == "product": return np.prod(np.maximum(x,EPS)**w, axis=1)
        if rule == "exp": return np.exp(x) @ w
        if rule == "tanh": return np.tanh(x) @ w
    p = 1.0 if parameters is None else float(np.asarray(parameters).ravel()[0])
    result = x[:,0]
    for j in range(1,x.shape[1]):
        result = _pair_tnorm(result, x[:,j], rule, p)
    return np.clip(result, 0, 1)


def fusion_bounds(rule: str, n_modalities: int) -> tuple[np.ndarray, np.ndarray]:
    if rule in RULES[:4]:
        return np.r_[np.full(n_modalities,0.001), 0.], np.r_[np.ones(n_modalities), 3.]
    if rule in ("einstein", "hamacher"):
        return np.array([0.]), np.array([1.])
    return np.array([0.05, 0.]), np.array([5., 1.])


def decode_particle(rule: str, position: np.ndarray, n_modalities: int):
    if rule in RULES[:4]:
        return position[:n_modalities], float(position[-1])
    if rule in ("einstein", "hamacher"):
        return None, float(position[-1])
    return position[:1], float(position[-1])
