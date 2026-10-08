
import json
from pathlib import Path
import numpy as np
from apso_auth.evaluation import train_and_evaluate
from apso_auth.fusion import fuse
from apso_auth.metrics import global_error

def run():
    output=[]
    for seed in (7,21,42):
        rng=np.random.default_rng(seed)
        labels=rng.integers(0,2,400)
        scores=np.clip(rng.normal(.30+.40*labels[:,None],.22,size=(400,2)),0,1)
        order=rng.permutation(400)
        learning,testing=order[:200],order[200:]
        for cfa in (.5,1.,1.5):
            selected,error=train_and_evaluate(scores[learning],labels[learning],scores[testing],labels[testing],cfa=cfa,swarm_size=20,iterations=15,seed=seed)
            baseline=global_error(labels[testing],fuse(scores[testing],"sum"),.5,cfa)
            output.append(dict(seed=seed,cfa=cfa,selected_rule=selected.rule,threshold=round(float(selected.threshold),6),optimized_e=round(float(error),6),fixed_sum_e=round(float(baseline),6),delta_baseline_minus_optimized=round(float(baseline-error),6)))
    results={"data_source":"generated synthetic scores, NOT HMOG","note":"fixed sum baseline is NOT reference [24] or [25]","configuration":{"seeds":[7,21,42],"samples_per_seed":400,"fusion_learning_samples":200,"fusion_evaluation_samples":200,"modalities":2,"swarm_size":20,"iterations":15},"runs":output}
    path=Path("reports/benchmark_results.json");path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(results,indent=2)+"\n")
    print(path)
    for level in (.5,1.,1.5):
        rows=[r for r in output if r["cfa"]==level]
        print(level,round(np.mean([r["optimized_e"] for r in rows]),4),round(np.mean([r["fixed_sum_e"] for r in rows]),4))
if __name__=="__main__":run()
