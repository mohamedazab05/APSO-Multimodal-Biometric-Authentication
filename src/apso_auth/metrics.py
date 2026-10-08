"""Biometric FAR/FRR and global cost metric (paper Eq. 2)."""
import numpy as np


def error_rates(labels, scores, threshold):
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not np.any(y) or not np.any(~y):
        raise ValueError("Need same-length arrays containing genuine and impostor examples")
    accepted = s >= threshold
    far = np.mean(accepted[~y]); frr = np.mean(~accepted[y])
    return float(far), float(frr)


def global_error(labels, scores, threshold, cfa):
    if not 0 <= cfa <= 2: raise ValueError("CFA must be in [0,2]")
    far, frr = error_rates(labels, scores, threshold)
    return cfa * far + (2-cfa) * frr
