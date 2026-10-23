from pathlib import Path
import argparse
import joblib,numpy as np
from common import load_signals,write
from train_v171 import predict_artifact

def main(a):
    meta,records=load_signals(a.data,a.ids);model=joblib.load(a.artifact)
    subjects=[r for r in meta['subjects'] if a.ids is None or r['subject_id'] in set(a.ids)]
    p=predict_artifact(model,records,a.device)
    names=[t['name'] for t in model['schema']]
    if p.shape!=(len(subjects),len(names)) or not np.isfinite(p).all():raise AssertionError('Invalid inference output')
    target=Path(a.output)
    if target.exists():raise FileExistsError(target)
    write(target,dict(questionnaire_labels_read=False,model=str(a.artifact),
      training_overlap_ids=[r['subject_id'] for r in subjects if r['subject_id'] in model.get('training_ids',[])],
      predictions={r['subject_id']:dict(zip(names,map(float,row))) for r,row in zip(subjects,p)},
      interpretation='Continuous questionnaire score estimates, not diagnoses.'))
    print(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--artifact',required=True)
    p.add_argument('--ids',nargs='+');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');main(p.parse_args())
