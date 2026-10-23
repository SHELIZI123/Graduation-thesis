from pathlib import Path
import argparse,csv,collections
import numpy as np
import joblib
from common import read,write,csvwrite,load_data,score,sha,new_dir
from train_v171 import predict_artifact
from preprocessing import target_stats
from features_v171 import FOCUS

def collect_runs(paths):
    protocols=[read(Path(p)/'protocol.json') for p in paths];p=protocols[0]
    for q in protocols[1:]:
        for key in ['version','data_manifest_sha256','plan','seeds','maximum_epochs','patience','cap','recipes','profile','foundation','code_sha256']:
            if q[key]!=p[key]:raise ValueError('Incompatible run: '+key)
    folds={}
    for root in map(Path,paths):
        for folder in sorted(root.glob('fold_*')):
            if not (folder/'complete.json').is_file():raise ValueError('Incomplete fold '+str(folder))
            c=read(folder/'complete.json');fi=c['fold']
            if fi in folds:raise ValueError('Duplicate fold')
            folds[fi]=folder
    return p,folds

def main(a):
    protocol,folders=collect_runs(a.runs);meta,records,y=load_data(a.data)
    if sha(Path(a.data)/'manifest.json')!=protocol['data_manifest_sha256']:raise ValueError('Wrong data')
    if not a.allow_partial and sorted(folders)!=list(range(5)):raise ValueError('Full fivefold results required')
    out=new_dir(a.output,[a.data,Path(__file__).parent]);schema=meta['schema'];lookup={t['name']:j for j,t in enumerate(schema)}
    rows=[];replay_count=0;maximum=0.;eligible=set()
    for fi,folder in sorted(folders.items()):
        checkpoint=read(folder/'complete.json');fold=protocol['plan']['folds'][fi]
        tr,te=fold['train'],fold['test'];eligible.update(te)
        train_groups={meta['subjects'][i]['group_id'] for i in tr};test_groups={meta['subjects'][i]['group_id'] for i in te}
        assert not train_groups&test_groups
        for inner in fold['inner']:
            assert set(inner['train'])|set(inner['validation'])==set(tr)
            assert not {meta['subjects'][i]['group_id'] for i in inner['train']}&{meta['subjects'][i]['group_id'] for i in inner['validation']}
        for name,h in checkpoint['artifact_sha256'].items():assert sha(folder/name)==h
        assert sha(folder/'predictions.csv')==checkpoint['prediction_sha256']
        with (folder/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            rr=list(csv.DictReader(f))
        if a.replay:
            candidates=sorted({r['model'] for r in rr})
            for name in candidates:
                artifact=joblib.load(folder/(name+'.joblib'))
                assert artifact['training_ids']==checkpoint['train_ids']
                assert artifact['test_ids']==checkpoint['test_ids']
                p=predict_artifact(artifact,[records[i] for i in te],a.device)
                predictions={meta['subjects'][i]['subject_id']:p[k] for k,i in enumerate(te)}
                for r in rr:
                    if r['model']==name:
                        diff=abs(predictions[r['subject_id']][lookup[r['target']]]-float(r['prediction']))
                        maximum=max(maximum,float(diff));replay_count+=1
                        if diff>1e-4:raise AssertionError('Replay mismatch '+name)
        mapping={meta['subjects'][i]['subject_id']:i for i in te}
        for r in rr:
            assert r['subject_id'] in mapping
            i=mapping[r['subject_id']];j=lookup[r['target']]
            assert float(r['y_true'])==y[i,j] and r['group_id']==meta['subjects'][i]['group_id']
            rows.append({**r,'fold':fi,'y_true':float(r['y_true']),'prediction':float(r['prediction']),
                         'complete_three_modality':r['complete_three_modality']=='True'})
    keys=[(r['model'],r['subject_id'],r['target']) for r in rows];assert len(keys)==len(set(keys))
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['model'],r['target'])].append(r)
    metrics=[];complete=[]
    for (name,target),rr in sorted(grouped.items()):
        expected=int(np.isfinite(y[sorted(eligible),lookup[target]]).sum());assert len(rr)==expected
        metrics.append(dict(model=name,target=target,**score([r['y_true'] for r in rr],[r['prediction'] for r in rr])))
        ss=[r for r in rr if r['complete_three_modality']]
        if ss:complete.append(dict(model=name,target=target,**score([r['y_true'] for r in ss],[r['prediction'] for r in ss])))
    contrasts=[('ridge_trimodal','stack_l1','all'),('mil_mean','mil_attention_pcgrad','all'),
               ('mil_attention','mil_attention_pcgrad','all'),('ridge_trimodal','tab_ensemble','all'),
               ('ridge_eeg_eye','ridge_trimodal','complete')]
    actual=[(l,r,s,t) for l,r,s in contrasts for t in FOCUS if (l,t) in grouped and (r,t) in grouped]
    paired=[];rng=np.random.default_rng(20261021)
    for left,right,scope,target in actual:
        l={r['subject_id']:r for r in grouped[(left,target)] if scope!='complete' or r['complete_three_modality']}
        r={r['subject_id']:r for r in grouped[(right,target)] if scope!='complete' or r['complete_three_modality']}
        ids=sorted(set(l)&set(r));gg=collections.defaultdict(list)
        for s in ids:gg[l[s]['group_id']].append(abs(l[s]['y_true']-l[s]['prediction'])-abs(r[s]['y_true']-r[s]['prediction']))
        sums=np.array([sum(z) for z in gg.values()]);counts=np.array([len(z) for z in gg.values()])
        draw=rng.integers(0,len(sums),size=(a.bootstrap,len(sums)));boot=sums[draw].sum(1)/counts[draw].sum(1)
        alpha=.05/max(len(actual),1)
        paired.append(dict(reference=left,candidate=right,target=target,scope=scope,n=len(ids),
            mae_gain=float(sums.sum()/counts.sum()),conditional_ci_low=float(np.quantile(boot,.025)),
            conditional_ci_high=float(np.quantile(boot,.975)),
            bonferroni_conditional_low=float(np.quantile(boot,alpha/2)),
            bonferroni_conditional_high=float(np.quantile(boot,1-alpha/2)),contrast_family_size=len(actual),
            fixed_oof_only=True,includes_retraining_uncertainty=False))
    csvwrite(out/'metrics.csv',metrics);csvwrite(out/'predictions.csv',rows);csvwrite(out/'complete_case_metrics.csv',complete)
    if paired:csvwrite(out/'paired_comparisons.csv',paired)
    write(out/'verification.json',dict(passed=True,complete_outer_folds=sorted(folders),full_fivefold=sorted(folders)==list(range(5)),
       profile=protocol['profile'],predictions_replayed=replay_count,maximum_replay_difference=maximum,
       smoke_is_not_efficacy_evidence=protocol['profile']=='smoke',group_disjoint=True,independent_test=False))
    write(out/'summary.json',dict(metrics=metrics,full_fivefold=sorted(folders)==list(range(5)),profile=protocol['profile'],
        exploratory=True,independent_test=False,seeds=protocol['seeds'],source_runs=[str(p) for p in a.runs]))
    mm={(r['model'],r['target']):r for r in metrics}
    names=['training_median','ridge_mask_only','ridge_trimodal',*(['tabpfn_v2'] if protocol.get('foundation') else []),*protocol['recipes'],'stack_l1','target_selector']
    lines=['# V17.1 结果','',f"实际完成外层折：{sorted(folders)}，运行配置：{protocol['profile']}。",'',
      '| 目标 | '+ ' | '.join(names)+' |','|---|'+'---:|'*len(names)]
    for target in FOCUS:
        lines.append('| '+target+' | '+' | '.join(f"{mm[(n,target)]['mae']:.4f}" for n in names)+' |')
    lines+=['','表格为原评分 MAE。全部 27 个输出的 MAE、RMSE、R²、Pearson 见 metrics.csv。',
      '初始化种子在每个训练折内先集成，受试者只计一次。没有把三个初始化种子当三份独立样本。',
      '外层划分沿用 V17。同一队列已经参与开发，外层结果仍是开发性比较，不能作为最终独立测试。',
      'stack_l1 权重只依据外层训练者的内层 OOF 预测拟合。内层误差参与选择，不能视为无偏成绩。',
      '配对区间基于固定 OOF 的身份组 bootstrap，同时给出预设比较族的 Bonferroni 条件区间。',
      '区间不包含模型重训、历史搜索和未完成的身份/访视核验不确定性。',
      '若 profile=smoke，以上分数仅用于排查程序，不能用于判断改善或论文结论。']
    (out/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Evaluation complete:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--allow-partial',action='store_true');p.add_argument('--replay',action='store_true')
    p.add_argument('--device',default='cpu');p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
