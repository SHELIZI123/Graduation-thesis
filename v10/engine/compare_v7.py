"""Compare fixed matching outer predictions; never use this report to select a model."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from common import load_json,csv_rows,save_json,save_csv,new_output,metrics

PAIRS=[('power_ridge','power_pca'),('cbramod_ridge','regional_cbramod'),
       ('eye_svr','eye_regularized'),('kernel_fusion','late_fusion'),
       ('kernel_fusion','guarded_fusion'),('concat_ridge','late_fusion'),
       ('concat_ridge','guarded_fusion'),('demographics_ridge','demographics_control')]


def paired_gain(gains,groups,seed=20261015):
    gains=np.asarray(gains,float);groups=np.asarray(groups);unique=np.unique(groups)
    # Preserve alias/visit groups as the sampling units, not windows.
    sums=np.array([gains[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
    sample=np.random.default_rng(seed).integers(len(unique),size=(2000,len(unique)))
    boot=sums[sample].sum(1)/counts[sample].sum(1)
    return float(gains.mean()),np.percentile(boot,[2.5,97.5]).tolist()


def main(args):
    if len(args.old)!=len(args.new):raise ValueError('Match old/new tasks explicitly')
    out=new_output(args.output,*[Path(p).resolve() for p in args.old+args.new])
    text=['# V8 改进与同划分验证','',
        '保留原始数据、V7 代码和 V7 结果。所有目标均为个人级量表回归，不是疾病诊断分类。',
        'V8 是查看 V7 结果后设计的改进，重复使用的外层划分仍属于开发期比较；最终结论需要新受试者或预留测试集。',
        '95% 区间为按受试者 group 配对 bootstrap 的描述性区间，未校正训练集重叠、多次比较和开发选择。','']
    comparisons=[];all_metrics=[];fallbacks={}
    for oldpath,newpath in zip(args.old,args.new):
        old,new=Path(oldpath),Path(newpath);a,b=load_json(old/'summary.json'),load_json(new/'summary.json')
        if not a['complete_outer_cv'] or not b['complete_outer_cv'] or a['plan_sha256']!=b['plan_sha256']:
            raise ValueError('Not complete matching frozen plans')
        ar,br=csv_rows(old/'predictions.csv'),csv_rows(new/'predictions.csv')
        decisions=load_json(new/'selections.json')
        for target,scores in b['results'].items():
            text += [f'## {target}','',
                '| 版本/模型 | 人数 | MAE ↓ | RMSE ↓ | R² ↑ |','|---|---:|---:|---:|---:|']
            for version,source in [('V7',a['results'][target]),('V8',scores)]:
                for name,m in source.items():
                    all_metrics.append({'target':target,'version':version,'model':name,**{k:m[k] for k in ['n','mae','rmse','r2']}})
                    text.append(f"| {version}/{name} | {m['n']} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |")
            reference=next(iter(scores.values()))
            for label in ['mean','median']:
                m=reference[label+'_baseline']
                text.append(f"| 训练集{label} | {reference['n']} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |")
            text += ['','| V7 → V8 | MAE 改善（正值为好） | 描述性95%区间 |','|---|---:|---|']
            for oname,nname in PAIRS:
                rr={(r['subject_id'],r['fold']):r for r in ar if r['target']==target and r['model']==oname}
                ss={(r['subject_id'],r['fold']):r for r in br if r['target']==target and r['model']==nname}
                if set(rr)!=set(ss):raise ValueError('Subject/fold coverage mismatch')
                gains=[];groups=[]
                for k,r in rr.items():
                    s=ss[k]
                    if r['y_true']!=s['y_true'] or r['group_id']!=s['group_id']:raise ValueError('Paired label/group mismatch')
                    truth=float(r['y_true']);gains.append(abs(truth-float(r['prediction']))-abs(truth-float(s['prediction'])))
                    groups.append(r['group_id'])
                gain,ci=paired_gain(gains,groups)
                comparisons.append({'target':target,'v7':oname,'v8':nname,'mae_gain':gain,'descriptive_95pct':ci})
                text.append(f'| {oname} → {nname} | {gain:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |')
            selected=[d['selected']['guarded_fusion']['spec'] for d in decisions if d['target']==target]
            counts=Counter('constant' if s['kind']=='constant' else 'blend' if s['kind']=='blend' else s['block'] for s in selected)
            fallbacks[target]=dict(counts)
            text += ['',f'guarded_fusion 五折选择：{dict(counts)}。',
                '回退减少误差时，只能解释为减少过拟合；常数回退不构成生理信号有效性的证据。','']
    text += ['## 如何使用这些结果','',
        '优先看生理信号模型是否稳定超过训练集均值/中位数，并检查各折差异。年龄/性别只在单列的人口学对照中使用。',
        '若优势很小、R² 接近零或区间跨零，不能宣称已经得到可靠预测器。不要按结果删人、换随机种子、反复用外层测试集挑模型。',
        '本版仍未解决同访视核对、EEG 事件含义、眼动任务和坐标定义这些数据层面的未知项；也没有补造瞳孔或同步窗口。',
        '新增未标注 EEG 可用于另行设计的自监督预训练，但不自动增加有标签评估样本。补标签、采集协议和前瞻验证比扩大调参网格更有价值。','']
    (out/'改进结果.md').write_text('\n'.join(text),encoding='utf-8')
    save_json(out/'paired_comparisons.json',comparisons);save_json(out/'fallback_counts.json',fallbacks)
    save_csv(out/'all_metrics.csv',all_metrics);print(out/'改进结果.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',nargs='+',required=True)
    p.add_argument('--new',nargs='+',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
