"""Held-out fusion training/evaluation and paired user-level bootstrap."""
import numpy as np
from .fusion import fuse
from .metrics import global_error
from .optimizer import search_best


def train_and_evaluate(x_learning, y_learning, x_eval, y_eval, cfa=1., **options):
    """Caller must ensure disjoint samples and no SVM training leakage."""
    result = search_best(x_learning, y_learning, cfa, **options)
    predicted = fuse(x_eval, result.rule, result.parameters)
    error = global_error(y_eval, predicted, result.threshold, cfa)
    return result, float(error)


def paired_user_bootstrap(labels, user_ids, proposed_scores, baseline_scores,
                          proposed_threshold, baseline_threshold, cfa=1.,
                          iterations=2000, seed=42):
    """CI for E_baseline - E_proposed using paired fixed-threshold scores."""
    y=np.asarray(labels); ids=np.asarray(user_ids)
    a=np.asarray(proposed_scores); b=np.asarray(baseline_scores)
    if any(len(z)!=len(y) for z in (ids,a,b)):
        raise ValueError("Input lengths must match")
    users=np.unique(ids); groups=[np.flatnonzero(ids==u) for u in users]
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(iterations):
        chosen=rng.integers(0,len(users),size=len(users))
        ind=np.concatenate([groups[j] for j in chosen])
        if np.all(y[ind]) or not np.any(y[ind]):
            continue
        diffs.append(global_error(y[ind],b[ind],baseline_threshold,cfa)-
                     global_error(y[ind],a[ind],proposed_threshold,cfa))
    if len(diffs)<max(50,iterations//2):
        raise ValueError("Too few valid resamples; check label balance")
    point=(global_error(y,b,baseline_threshold,cfa)-global_error(y,a,proposed_threshold,cfa))
    ci=np.quantile(diffs,[0.025,0.975])
    return {"difference":float(point), "ci_lower":float(ci[0]),
            "ci_upper":float(ci[1]),"valid_resamples":len(diffs)}
