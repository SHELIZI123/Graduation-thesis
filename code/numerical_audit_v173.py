"""Audit real TabPFN repeat/replay and quantify (not assume) CPU/GPU agreement."""
from pathlib import Path
import argparse,copy
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from features_v172 import load_inputs
from foundation_v173 import predict_context,runtime
from common import write

def main(a):
    meta,records,y=load_inputs(a.data,a.extras);model=joblib.load(a.model)
    ids=model['prediction_query_snapshot']['subject_ids'];byid={r['subject_id']:r for r in records};rr=[byid[i] for i in ids]
    first,d1=predict_context(model,rr,a.device,replay=True)
    second,d2=predict_context(model,rr,a.device,replay=True)
    repeated=float(abs(first-second).max());assert repeated<=1e-4
    computed,d3=predict_context(copy.deepcopy(model),rr,a.device,replay=False)
    recomputed=float(abs(first-computed).max());assert recomputed<=1e-4
    for change in ['ids','tensor']:
        bad=copy.deepcopy(model)
        if change=='ids':bad['prediction_query_snapshot']['subject_ids'][0]='wrong'
        else:bad['prediction_query_snapshot']['matrix'][0,0]+=1
        try:predict_context(bad,rr,a.device,replay=True)
        except ValueError:pass
        else:raise AssertionError('Invalid saved query was accepted: '+change)
    cpu,dc=predict_context(model,rr,'cpu',replay=True)
    per_target={t['name']:float(abs(first[:,j]-cpu[:,j]).max()) for j,t in enumerate(model['schema']) if any(v['target']==t['name'] for v in d1['precision'])}
    result=dict(state='passed',scope='local engineering only',runtime=runtime(),device=a.device,
        repeat_max_difference=repeated,recomputed_query_prediction_max_difference=recomputed,
        saved_query_recomputation=d1,query_id_and_tensor_tamper_rejected=True,
        cpu_gpu_max_difference=float(abs(first-cpu).max()),cpu_gpu_max_difference_by_target=per_target,
        cross_platform_equality_guaranteed=False)
    write(a.output,result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--extras',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
