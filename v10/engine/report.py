"""Read-only comparisons: identical cohorts/plans; cluster bootstrap by person."""
import argparse
from pathlib import Path
import numpy as np
from common import load_json,csv_rows,save_json,new_output,metrics


def main(args):
    roots=[Path(p).resolve(strict=True) for p in args.runs]
    out=new_output(args.output,*roots)
    plans={load_json(p/'summary.json')['plan_sha256'] for p in roots}
    if len(plans)!=1:raise ValueError('Different cohort/split plans: cannot directly compare')
    reference=None;reference_truth=None;report={}
    for path in roots:
        rows=csv_rows(path/'predictions.csv')
        keys=[(r['subject_id'],r['target'],r['fold']) for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate held-out predictions')
        truth_map={k:float(r['y_true']) for k,r in zip(keys,rows)}
        if reference is None:reference=set(keys);reference_truth=truth_map
        if set(keys)!=reference:raise ValueError('Different held-out subjects/targets/folds')
        if truth_map!=reference_truth:raise ValueError('Held-out labels differ across runs')
        summary={}
        for t in sorted({r['target'] for r in rows}):
            rr=sorted([r for r in rows if r['target']==t],key=lambda r:r['subject_id'])
            y=np.array([float(r['y_true']) for r in rr]);p=np.array([float(r['prediction']) for r in rr])
            b=np.array([float(r['mean_prediction']) for r in rr]);g=np.array([r['group_id'] for r in rr])
            gains=np.abs(y-b)-np.abs(y-p);groups=np.unique(g);rng=np.random.default_rng(20261013)
            cluster=[gains[g==k] for k in groups];boot=[]
            for _ in range(2000):
                sample=rng.integers(len(groups),size=len(groups))
                boot.append(float(np.concatenate([cluster[i] for i in sample]).mean()))
            summary[t]={'model':metrics(y,p),'mean_baseline':metrics(y,b),'mae_gain_vs_mean':float(gains.mean()),
                'descriptive_cluster_bootstrap_95pct':np.percentile(boot,[2.5,97.5]).tolist(),
                'n_records':len(y),'n_person_groups':len(groups)}
        report[path.name]=summary
    save_json(out/'comparison.json',{'runs':report,'same_plan':True,
              'uncertainty_note':'Descriptive only: fold models share training data; not independent clinical validation. No automatic model selection.'})
    print('Comparison saved:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
