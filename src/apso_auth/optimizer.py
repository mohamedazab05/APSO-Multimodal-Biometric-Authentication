"""Adaptive PSO with diversity heuristic and elitist perturbation.

Independent research implementation inspired by the paper; not verified as
equivalent to the authors' original APSO implementation.
"""
from dataclasses import dataclass
import numpy as np
from .fusion import RULES, fuse, fusion_bounds, decode_particle
from .metrics import global_error

@dataclass
class SearchResult:
    rule: str
    parameters: np.ndarray | None
    threshold: float
    learning_error: float


def _search_rule(scores, labels, cfa, rule, *, swarm_size, iterations, seed):
    rng = np.random.default_rng(seed)
    lower, upper = fusion_bounds(rule, scores.shape[1]); dim = len(lower)
    pos = rng.uniform(lower, upper, size=(swarm_size,dim))
    vel = np.zeros_like(pos)
    def objective(v):
        weights, eta = decode_particle(rule, v, scores.shape[1])
        return global_error(labels, fuse(scores, rule, weights), eta, cfa)
    fitness = np.array([objective(v) for v in pos])
    best_pos = pos.copy(); best_fit = fitness.copy()
    gi = int(np.argmin(best_fit)); gpos = best_pos[gi].copy(); gfit = best_fit[gi]
    for it in range(iterations):
        diversity = np.mean(np.std((pos-lower)/(upper-lower), axis=0))
        omega = np.clip(0.45 + 0.45 * diversity / 0.3, 0.45, 0.9)
        c1 = np.clip(1.5 + diversity, 1., 2.)
        c2 = np.clip(2.0 - diversity, 1., 2.)
        vel = omega*vel + c1*rng.random(pos.shape)*(best_pos-pos) + c2*rng.random(pos.shape)*(gpos-pos)
        pos = np.clip(pos+vel, lower, upper)
        fitness = np.array([objective(v) for v in pos])
        improved = fitness < best_fit
        best_fit[improved] = fitness[improved]; best_pos[improved] = pos[improved]
        gi = int(np.argmin(best_fit))
        if best_fit[gi] < gfit:
            gfit = best_fit[gi]; gpos = best_pos[gi].copy()
        if diversity < 0.08:
            candidate = gpos.copy(); k = rng.integers(dim)
            candidate[k] = np.clip(candidate[k]+rng.normal(0,0.08*(upper[k]-lower[k])),lower[k],upper[k])
            val = objective(candidate)
            if val < gfit: gfit = val; gpos = candidate
    weights, eta = decode_particle(rule, gpos, scores.shape[1])
    return SearchResult(rule, weights, eta, float(gfit))


def search_best(scores, labels, cfa, swarm_size=100, iterations=100, seed=42, rules=RULES):
    if swarm_size < 2 or iterations < 1: raise ValueError("Invalid optimization settings")
    candidates = [_search_rule(scores, labels, cfa, rule, swarm_size=swarm_size,
                               iterations=iterations, seed=seed+i)
                  for i,rule in enumerate(rules)]
    return min(candidates, key=lambda result: result.learning_error)
