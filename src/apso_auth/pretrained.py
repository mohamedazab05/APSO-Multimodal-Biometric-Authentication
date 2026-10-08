"""Load portable linear SVM model parameters for calibrated scoring."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit

def predict_probability(features, model_path):
    """Return probability for positive class from a JSON SVM checkpoint."""
    model=json.loads(Path(model_path).read_text())
    features=np.asarray(features,dtype=float)
    if features.ndim != 2 or features.shape[1] != model["features"]:
        raise ValueError(f"Expected (n,{model['features']}) feature matrix")
    normalized=(features-np.asarray(model["mean"]))/np.asarray(model["scale"])
    decision=normalized@np.asarray(model["linear_coef"])+float(model["intercept"][0])
    return expit(-(float(model["probA"][0])*decision+float(model["probB"][0])))
