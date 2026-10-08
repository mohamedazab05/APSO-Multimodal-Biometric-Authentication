import numpy as np
from apso_auth.fusion import RULES, fuse
from apso_auth.metrics import error_rates, global_error
from apso_auth.optimizer import search_best
from apso_auth.evaluation import paired_user_bootstrap

def test_fusion_rules_valid():
    x=np.array([[0.,0.],[0.1,0.4],[0.8,0.9],[1.,1.]])
    for rule in RULES:
        y=fuse(x,rule)
        assert np.all(np.isfinite(y)),rule
        assert len(y)==len(x)
        if rule != 'exp': assert np.all(y>=-1e-9) and np.all(y<=1+1e-9)

def test_identity_bounds():
    x=np.array([[0.,0.],[1.,1.]])
    for rule in RULES:
        out=fuse(x,rule)
        if rule not in ('exp','tanh'):
            assert np.allclose(out,[0,1],atol=1e-6), (rule,out)
        if rule == 'tanh': assert np.allclose(out,[0,np.tanh(1)]), (rule,out)

def test_error():
    y=[0,0,1,1]; s=[0.2,0.8,0.6,0.9]
    assert error_rates(y,s,0.5)==(0.5,0.)
    assert global_error(y,s,0.5,1)==0.5

def test_search_and_bootstrap():
    rng=np.random.default_rng(7)
    y=np.r_[np.zeros(30),np.ones(30)]; x=np.clip(rng.normal(0.3+0.4*y[:,None],.2,(60,2)),0,1)
    result=search_best(x,y,1,swarm_size=6,iterations=3,rules=['sum','einstein'])
    assert 0<=result.learning_error<=2
    a=np.linspace(0,1,60); b=np.linspace(0.1,.9,60)
    res=paired_user_bootstrap(y,np.repeat(np.arange(20),3),a,b,.5,.5,iterations=100,seed=2)
    assert res['ci_lower']<=res['ci_upper']
