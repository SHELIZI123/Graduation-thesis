from pathlib import Path
import argparse
import joblib,torch
import numpy as np
from threadpoolctl import threadpool_limits
from common import load_signals,write,read,sha
from engine import predict

def main(a):
    if Path(a.output).exists():raise FileExistsError(a.output)
    meta,records=load_signals(a.data,a.ids)
    order=[s['subject_id'] for s in meta['subjects'] if s['subject_id'] in set(a.ids)]
    mm=read(Path(a.models)/'manifest.json');path=Path(a.models)/(a.model+'.joblib')
    if sha(path)!=mm['models_sha256'][a.model]:raise ValueError('Changed model file')
    model=joblib.load(path);p=predict(model,records,a.device)
    values={sid:{t['name']:float(p[i,j]) for j,t in enumerate(meta['schema'])} for i,sid in enumerate(order)}
    write(a.output,{'model':a.model,'predictions':values,'training_overlap':sorted(set(order)&set(mm['training_ids'])),
                    'questionnaire_labels_read':False,'disease_diagnoses_produced':False,'exploratory':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--models',required=True)
    p.add_argument('--model',default='set_ple');p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):main(a)
