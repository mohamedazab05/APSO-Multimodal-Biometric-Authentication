"""Modality-specific SVM scoring from pre-extracted feature matrices.

Does not implement the paper's unspecified HMOG preprocessing/features.
Input matrices must contain numeric engineered features aligned by sample ID.
"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fit_modality_svms(train_features: dict, train_labels, *, c=1.0, gamma='scale', seed=42):
    labels=np.asarray(train_labels).astype(int)
    if set(np.unique(labels)) != {0,1}:
        raise ValueError('Training data must include genuine and impostor samples')
    models={}
    for modality,x in train_features.items():
        x=np.asarray(x,float)
        if x.ndim!=2 or len(x)!=len(labels) or not np.all(np.isfinite(x)):
            raise ValueError(f'Invalid features for {modality}')
        model=make_pipeline(StandardScaler(), SVC(C=c,gamma=gamma,probability=True,random_state=seed))
        model.fit(x,labels)
        models[modality]=model
    return models


def score_modality_svms(models: dict, features: dict):
    modalities=sorted(models)
    if set(modalities)!=set(features): raise ValueError('Modality sets differ')
    cols=[]
    for m in modalities:
        x=np.asarray(features[m],float)
        if x.ndim!=2 or not np.all(np.isfinite(x)):
            raise ValueError(f'Invalid features for {m}')
        probabilities=models[m].predict_proba(x)
        genuine_index=list(models[m].classes_).index(1)
        cols.append(probabilities[:,genuine_index])
    return np.column_stack(cols),modalities
