"""Merge, validate, replay and summarize subject-level out-of-fold predictions."""
from pathlib import Path
import argparse,csv,collections
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
from common import read,write,csvwrite,sha,new_dir,score
from features_v172 import load_inputs
from features_v171 import FOCUS
from engine_v173 import predict_candidate

def check_anchor(selection,allowed,groups):
    assert set(selection['fit_ids'])==set(allowed)
    for fold in selection['folds']:
        tr,va=set(fold['train_ids']),set(fold['validation_ids'])
        assert tr|va==set(allowed) and not tr&va
        assert not {groups[i] for i in tr}&{groups[i] for i in va}

def collect(paths):
    protocols=[read(Path(p)/'protocol.json') for p in paths];first=protocols[0];folders={}
    for p in protocols[1:]:
        for k in first:
            if k!='hardware' and first[k]!=p[k]:raise ValueError('Worker protocol differs: '+k)
    for root in map(Path,paths):
        if read(root/'status.json')['state']!='complete':raise ValueError('Worker unfinished')
        for folder in root.glob('fold_*'):
            c=read(folder/'complete.json');fi=c['fold']
            if fi in folders:raise ValueError('Duplicate fold')
            folders[fi]=folder
    return first,folders

def main(a):
    protocol,folders=collect(a.runs);meta,records,y=load_inputs(a.data,a.extras)
    if {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}!=protocol['code_sha256']:raise ValueError('Evaluation source differs from training')
    if sha(Path(a.data)/'manifest.json')!=protocol['data_manifest_sha256'] or sha(Path(a.extras)/'manifest.json')!=protocol['extras_manifest_sha256']:raise ValueError('Wrong inputs')
    if not a.allow_partial and sorted(folders)!=list(range(5)):raise ValueError('Five folds required')
    out=new_dir(a.output,[a.data,a.extras,Path(__file__).parent]);schema=meta['schema'];lookup={t['name']:j for j,t in enumerate(schema)}
    rows=[];maximum=0.;count=0;eligible=set();decisions={};query_audits=[]
    group_lookup={r['subject_id']:r['group_id'] for r in records}
    for fi,folder in sorted(folders.items()):
        c=read(folder/'complete.json');f=protocol['plan']['folds'][fi];tr,te=f['train'],f['test'];eligible.update(te)
        train_ids=[meta['subjects'][i]['subject_id'] for i in tr];test_ids=[meta['subjects'][i]['subject_id'] for i in te]
        if c['train_ids']!=train_ids or c['test_ids']!=test_ids:raise ValueError('Fold IDs changed')
        groups=protocol['plan']['groups'];assert not {groups[i] for i in tr}&{groups[i] for i in te}
        for inner in f['inner']:
            assert set(inner['train'])|set(inner['validation'])==set(tr)
            assert not {groups[i] for i in inner['train']}&{groups[i] for i in inner['validation']}
        assert sha(folder/'predictions.csv')==c['prediction_sha256']
        with (folder/'predictions.csv').open(encoding='utf-8-sig',newline='') as stream:rr=list(csv.DictReader(stream))
        assert {r['model'] for r in rr}==set(protocol['names'])
        assert set(c['artifact_sha256'])=={n+'.joblib' for n in protocol['names']}
        mapping={sid:i for sid,i in zip(test_ids,te)}
        for name in protocol['names']:
            path=folder/(name+'.joblib');assert sha(path)==c['artifact_sha256'][path.name]
            if a.replay:
                model=joblib.load(path);assert model['training_ids']==train_ids and model['test_ids']==test_ids
                if 'anchor_selection' in model:check_anchor(model['anchor_selection'],train_ids,group_lookup)
                if model['kind']=='v172_residual':
                    assert model['prep'].training_ids==train_ids
                    for fold in model['crossfit']:
                        assert set(fold['train_ids'])|set(fold['validation_ids'])==set(train_ids)
                        g={r['subject_id']:r['group_id'] for r in records}
                        assert not {g[i] for i in fold['train_ids']}&{g[i] for i in fold['validation_ids']}
                        check_anchor(fold['anchor_selection'],fold['train_ids'],group_lookup)
                    assert all(log['optimizer_steps']>0 and log['epochs']>0 for log in model['logs'])
                pred,detail=predict_candidate(model,[records[i] for i in te],a.device,replay=True,components=True)
                if model['kind']=='v173_foundation':query_audits.append(dict(fold=fi,model=name,**detail))
                pm={sid:p for sid,p in zip(test_ids,pred)}
                for r in rr:
                    if r['model']!=name:continue
                    difference=abs(float(pm[r['subject_id']][lookup[r['target']]])-float(r['prediction']))
                    maximum=max(maximum,difference);count+=1
                    if difference>1e-4+1e-5*abs(float(r['prediction'])):raise AssertionError('Prediction replay mismatch: '+name)
        for r in rr:
            i=mapping[r['subject_id']];j=lookup[r['target']]
            assert float(r['y_true'])==y[i,j] and r['group_id']==meta['subjects'][i]['group_id']
            assert int(r['fold'])==fi
            assert (r['complete_three_modality']=='True')==all(meta['subjects'][i]['modality_available'].values())
            assert (r['eeg_available']=='True')==meta['subjects'][i]['modality_available']['eeg']
            rows.append({**r,'fold':fi,'y_true':float(r['y_true']),'prediction':float(r['prediction']),
                         'complete_three_modality':r['complete_three_modality']=='True','eeg_available':r['eeg_available']=='True'})
        decisions[str(fi)]=read(folder/'decisions.json')
    keys=[(r['model'],r['subject_id'],r['target']) for r in rows];assert len(set(keys))==len(keys)
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['model'],r['target']].append(r)
    metrics=[]
    for (name,target),rr in sorted(grouped.items()):
        j=lookup[target];expected={meta['subjects'][i]['subject_id'] for i in eligible if np.isfinite(y[i,j])}
        assert {r['subject_id'] for r in rr}==expected
        for scope,ss in [('all',rr),('complete',[r for r in rr if r['complete_three_modality']]),('eeg_available',[r for r in rr if r['eeg_available']])]:
            if ss:metrics.append(dict(model=name,target=target,scope=scope,**score([r['y_true'] for r in ss],[r['prediction'] for r in ss])))
    contrasts=[tuple(c) for c in protocol['contrast_spec']]
    pairs=[];rng=np.random.default_rng(20261024);family=len(contrasts)*len(FOCUS)
    for reference,name,scope in contrasts:
        for target in FOCUS:
            left={r['subject_id']:r for r in grouped[reference,target] if scope=='all' or r[scope]};right={r['subject_id']:r for r in grouped[name,target] if scope=='all' or r[scope]}
            bygroup=collections.defaultdict(list)
            for sid in left:
                l=left[sid];r=right[sid];bygroup[l['group_id']].append(abs(l['y_true']-l['prediction'])-abs(r['y_true']-r['prediction']))
            sums=np.array([sum(v) for v in bygroup.values()]);nn=np.array([len(v) for v in bygroup.values()])
            draw=rng.integers(0,len(sums),size=(a.bootstrap,len(sums)));b=sums[draw].sum(1)/nn[draw].sum(1);alpha=.05/family
            pairs.append(dict(reference=reference,candidate=name,target=target,scope=scope,mae_gain=float(sums.sum()/nn.sum()),
                conditional_low=float(np.quantile(b,.025)),conditional_high=float(np.quantile(b,.975)),
                bonferroni_low=float(np.quantile(b,alpha/2)),bonferroni_high=float(np.quantile(b,1-alpha/2)),
                fixed_oof_only=True,includes_retraining_uncertainty=False,contrast_family_size=family))
    csvwrite(out/'predictions.csv',rows);csvwrite(out/'metrics.csv',metrics);csvwrite(out/'paired_comparisons.csv',pairs)
    write(out/'query_replay_audit.json',query_audits)
    write(out/'decisions.json',decisions);write(out/'verification.json',dict(state='passed',folds=sorted(folders),
        full_fivefold=sorted(folders)==list(range(5)),replayed_predictions=count,maximum_replay_difference=maximum,replay_tolerance='1e-4 + 1e-5 * abs(saved prediction); saved-query tensors only for TabPFN replay',
        observed_labels_verified=True,group_disjoint=True,source_protocol=protocol,
        statistical_note='Conditional fixed OOF bootstrap only; historical development/model-search uncertainty not included'))
    lines=['# V17.3 结果','',f"外层折：{sorted(folders)}；profile={protocol['profile']}。当前队列已用于开发，不是独立最终测试。",'',
        'MAE 越低越好。smoke 或部分折结果只验证工程，不用于判断模型优劣。', '',
        '| 模型 | '+' | '.join(FOCUS)+' |','|---|'+'---:|'*len(FOCUS)]
    table={(m['model'],m['target']):m['mae'] for m in metrics if m['scope']=='all'}
    for name in protocol['names']:lines.append('| '+name+' | '+' | '.join(f'{table[name,t]:.4f}' for t in FOCUS)+' |')
    lines+=['','残差零修正回退与 ridge_anchor 使用同一训练内选参算法；eeg_* 的表示对照应看 eeg_available 子集。',
        '完整子集、EEG 可用子集的相同受试者比较见 metrics.csv；条件区间见 paired_comparisons.csv。',
        '各折选中轮数、梯度、修正幅度与分支日志见 decisions.json。']
    (out/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Verified',len(folders),'folds, replay',count,'max difference',maximum,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--extras',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--replay',action='store_true');p.add_argument('--allow-partial',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--bootstrap',type=int,default=20000);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
