"""Synthetic-only smoke benchmark; not HMOG or paper results."""
import json
from pathlib import Path
import numpy as np
from apso_auth.evaluation import train_and_evaluate
from apso_auth.metrics import global_error
from apso_auth.fusion import fuse

def run():
    output=[]
    for seed in (7,21,42):
        rng=np.random.default_rng(seed)
        labels=rng.integers(0,2,400)
        scores=np.clip(rng.normal(.30+.40*labels[:,None],.22,size=(400,2)),0,1)
        idx=rng.permutation(400)
        learn,test=idx[:200],idx[200:]
        for cfa in (.5,1.,1.5):
            fit,proposed=train_and_evaluate(scores[learn],labels[learn],scores[test],labels[test],
                cfa=cfa,swarm_size=20,iterations=15,seed=seed)
            baseline=global_error(labels[test],fuse(scores[test],"sum"),.5,cfa)
            output.append(dict(seed=seed,cfa=cfa,selected_rule=fit.rule,
                threshold=round(float(fit.threshold),6),optimized_e=round(float(proposed),6),
                fixed_sum_e=round(float(baseline),6),
                delta_baseline_minus_optimized=round(float(baseline-proposed),6)))
    p=Path("reports/synthetic_benchmark_results.json")
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps({"note":"SYNTHETIC ONLY; fixed sum is NOT reference [24] or [25]",
       "configuration":{"seeds":[7,21,42],"samples_per_seed":400,
       "fusion_learning_samples":200,"fusion_evaluation_samples":200,
       "modalities":2,"swarm_size":20,"iterations":15},"runs":output},indent=2)+"\n")
    print(p)
    for row in output: print(row)
    for level in (.5,1.,1.5):
        rows=[x for x in output if x["cfa"]==level]
        print("CFA",level,"APSO",np.mean([x["optimized_e"] for x in rows]),
              "FIXED",np.mean([x["fixed_sum_e"] for x in rows]))
if __name__=="__main__": run()
