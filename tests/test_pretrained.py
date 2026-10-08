
from pathlib import Path
import numpy as np
from apso_auth.pretrained import predict_probability

ROOT=Path(__file__).resolve().parents[1]

def test_saved_svm_models_predict_probabilities():
    touch=np.array([[0.1,0.2,0.3],[0.6,0.7,0.8]],dtype=float)
    motion=np.array([[0.1,0.2,0.3,0.4],[0.6,0.7,0.8,0.9]],dtype=float)
    for name,x in (("touch",touch),("motion",motion)):
        scores=predict_probability(x,ROOT/"models"/f"{name}_svm.json")
        assert scores.shape==(2,)
        assert np.all(np.isfinite(scores))
        assert np.all((scores>=0)&(scores<=1))
