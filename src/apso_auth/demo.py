"""Synthetic demonstration. NEVER interpret output as paper experimental results."""
import numpy as np
from .evaluation import train_and_evaluate

def main():
    rng=np.random.default_rng(42); n=400
    y=rng.integers(0,2,size=n).astype(bool)
    x=np.clip(rng.normal(0.3+0.4*y[:,None],0.22,size=(n,2)),0,1)
    idx=rng.permutation(n); learning=idx[:200]; evaluation=idx[200:]
    print("SYNTHETIC DEMO — not publication results")
    for cfa in (0.5,1.,1.5):
        r,e=train_and_evaluate(x[learning],y[learning],x[evaluation],y[evaluation],
                               cfa=cfa,swarm_size=20,iterations=15,seed=42)
        print(f"CFA={cfa:.1f} selected={r.rule} threshold={r.threshold:.4f} held_out_E={e:.4f}")
if __name__ == "__main__": main()
