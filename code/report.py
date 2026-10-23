from pathlib import Path
import argparse,csv,collections
import numpy as np
from common import read,write,csvwrite,new_dir,score,load_data
from refit import merge_inputs
from preprocessing import target_stats

def main(a):
    protocol,decisions=merge_inputs(a.runs);meta,records,y=load_data(a.data)
    out=new_dir(a.output,[a.data,Path(__file__).parent]);rows=[]
    for path in a.runs:
        with (Path(path)/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                rows.append({**r,'fold':int(r['fold']),'y_true':float(r['y_true']),'prediction':float(r['prediction']),
                             'complete_three_modality':r['complete_three_modality']=='True'})
    keys=[(r['subject_id'],r['target'],r['model']) for r in rows]
    assert len(keys)==len(set(keys))
    ix={s['subject_id']:i for i,s in enumerate(meta['subjects'])}
    stats={f['fold']:target_stats(y[f['train']])[1] for f in protocol['plan']['folds']}
    jj={t['name']:j for j,t in enumerate(meta['schema'])};tt={t['name']:t for t in meta['schema']}
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['model'],r['target'])].append(r)
    metrics=[];complete=[];family=collections.defaultdict(list)
    for (model,target),rr in sorted(groups.items()):
        expected=int(np.isfinite(y[protocol['plan']['eligible_indices'],jj[target]]).sum())
        if not model.endswith('_probe'):assert len(rr)==expected
        metrics.append({'model':model,'target':target,'family':tt[target]['family'],**score([r['y_true'] for r in rr],[r['prediction'] for r in rr])})
        ss=[r for r in rr if r['complete_three_modality']]
        if ss:complete.append({'model':model,'target':target,'family':tt[target]['family'],**score([r['y_true'] for r in ss],[r['prediction'] for r in ss])})
        nm=float(np.mean([abs(r['prediction']-r['y_true'])/stats[r['fold']][jj[target]] for r in rr]))
        family[(model,tt[target]['family'])].append((nm,tt[target]['weight']))
    families=[{'model':model,'family':name,'weighted_normalized_mae':float(np.average([v for v,w in vv],weights=[w for v,w in vv]))}
              for (model,name),vv in sorted(family.items())]
    comparisons=[];rng=np.random.default_rng(20261024)
    for left,right,title in [('ridge_eeg_eye','ridge_trimodal','heart_increment_in_same_ridge_family'),
                             ('set_shared','set_ple','private_shared_expert_routing'),
                             ('ridge_trimodal','set_ple','deep_vs_linear')]:
        for target in jj:
            arows={r['subject_id']:r for r in groups.get((left,target),[]) if r['complete_three_modality']}
            brows={r['subject_id']:r for r in groups.get((right,target),[]) if r['complete_three_modality']}
            ids=sorted(set(arows)&set(brows))
            if len(ids)<10:continue
            gain=np.array([abs(arows[s]['prediction']-arows[s]['y_true'])-abs(brows[s]['prediction']-brows[s]['y_true']) for s in ids])
            group_gain=collections.defaultdict(list)
            for s,g in zip(ids,gain):group_gain[meta['subjects'][ix[s]]['group_id']].append(g)
            total=np.array([sum(v) for v in group_gain.values()]);n=np.array([len(v) for v in group_gain.values()])
            sampled=rng.integers(0,len(total),size=(5000,len(total)));boot=total[sampled].sum(1)/n[sampled].sum(1)
            comparisons.append({'comparison':title,'reference':left,'candidate':right,'target':target,'n':len(ids),
                                'mae_gain':float(gain.mean()),'conditional_ci_low':float(np.quantile(boot,.025)),
                                'conditional_ci_high':float(np.quantile(boot,.975)),
                                'scope':'same_complete_three_modality_subjects','fixed_oof_bootstrap_only':True,
                                'multiple_comparisons_corrected':False})
    csvwrite(out/'predictions.csv',rows);csvwrite(out/'metrics.csv',metrics)
    csvwrite(out/'complete_case_metrics.csv',complete);csvwrite(out/'family_metrics.csv',families)
    if comparisons:csvwrite(out/'paired_complete_case_comparisons.csv',comparisons)
    write(out/'summary.json',{'complete':True,'full_fivefold':True,'metrics':metrics,'family_metrics':families,
                            'data_summary':meta['summary'],'partial_labels_not_imputed':True,
                            'independent_test':False,'exploratory':True,'source_runs':[str(p) for p in a.runs]})
    focus=['psqi_total','scl_yy','scl_jl','scl_mean','epq_E','epq_N','epq_P','epq_L']
    models=['training_median','ridge_eeg_eye','ridge_heart','ridge_trimodal','set_ple','set_mmoe','set_shared']
    active=[m for m in models if any(r['model']==m for r in metrics)]
    lookup={(r['model'],r['target']):r for r in metrics}
    lines=['# V17 三模态与三份量表评分模型', '',
           f"完整五折已完成。参与监督训练 {len(protocol['plan']['eligible_indices'])} 人，评分目标 {len(jj)} 个。所有量表仅作为监督标签。",'',
           '| 评分目标 | 有标签人数 | '+' | '.join(active)+' |',
           '|---|---:|'+'---:|'*len(active)]
    for target in focus:
        n=lookup[(active[0],target)]['n'];vv=[f"{lookup[(m,target)]['mae']:.4f}" for m in active]
        lines.append('| '+target+' | '+str(n)+' | '+' | '.join(vv)+' |')
    lines+=['','表中为未取整的原始评分 MAE，越低越好。完整 27 项的 MAE、RMSE、R² 和 Pearson 见 metrics.csv。', '',
            '## 解释边界','',
            '- 这批样本与旧 181/175 人研究不同，不直接横向比较旧 MAE。新纳入样本已经用于开发。',
            '- 心率分支使用 PPG 形态、原生单位搏间期及相对变化。没有自行认定采样率或换算 BPM、ms、Hz HRV。',
            '- PSQI 七分项只有 85 人可用，其他人的分项保持缺失。EPQ 四维使用原始分，标准分不与原始分混合。',
            '- 量表组内先按训练目标标准差归一化，组间平衡，辅助目标降权。没有把人格分直接解释为疾病严重度。',
            '- 训练与验证分离缩放和早停，外层测试不决定训练轮数。epoch=0 是合法的训练均值检查点，见各折 decisions.json。',
            '- heart_removed_probe 是对同一个已训练网络屏蔽心率的敏感性检查，不等于重新训练的无心率消融。',
            '- 心率增益的主要对比是同一批三模态完整者上的 ridge_eeg_eye 与 ridge_trimodal。整个模型另有单模态与缺失掩码对照。',
            '- 配对区间是固定 OOF 的条件 bootstrap，未覆盖重新训练、历史搜索及多重比较。身份和访视仍未人工核验，结果保持开发性。',
            '- 结构来自 Deep Sets、HeMIS、PLE/MMoE 的适配，未宣称论文原创性或保证超过传统模型。']
    (out/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:16]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);main(p.parse_args())
