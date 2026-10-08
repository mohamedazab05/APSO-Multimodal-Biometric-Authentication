import numpy as np
from apso_auth.svm import fit_modality_svms, score_modality_svms

def test_svm_scores():
    rng=np.random.default_rng(13)
    y=np.tile([0,1],30)
    features={'touch':rng.normal(y[:,None],0.3,size=(60,3)),
              'motion':rng.normal(y[:,None],0.5,size=(60,4))}
    models=fit_modality_svms({k:v[:40] for k,v in features.items()},y[:40])
    scores,modalities=score_modality_svms(models,{k:v[40:] for k,v in features.items()})
    assert modalities==['motion','touch']
    assert scores.shape==(20,2)
    assert np.all((scores>=0)&(scores<=1))
