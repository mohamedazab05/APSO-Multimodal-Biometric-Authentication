"""Build touch and motion SVM checkpoints using generated example features.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
def train():
    rng=np.random.default_rng(2026)
    n=80
    y=np.tile([0,1],n//2)
    out=ROOT/"models"
    out.mkdir(exist_ok=True)
    for name,noise,d in [("touch",.55,3),("motion",.8,4)]:
        features=rng.normal(.35+.45*y[:,None],noise,(n,d))
        model=make_pipeline(StandardScaler(),SVC(kernel="linear",C=1,probability=True,random_state=2026))
        model.fit(features,y)
        scaler,svm=model.steps[0][1],model.steps[1][1]
        payload=dict(modality=name,model_type="SVC linear kernel with Platt scaling",
          training_source="generated example features (NOT HMOG)",seed=2026,features=d,
          mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),linear_coef=svm.coef_.ravel().tolist(),
          intercept=svm.intercept_.tolist(),probA=svm.probA_.tolist(),probB=svm.probB_.tolist(),
          classes=svm.classes_.tolist(),n_training_samples=n)
        path=out/f"{name}_svm.json";path.write_text(json.dumps(payload,indent=2)+"\n")
        print("Saved",path)
if __name__=="__main__":train()
