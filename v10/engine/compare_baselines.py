"""Descriptive paired group bootstrap; no automatic model/target selection."""
import argparse
from pathlib import Path
import numpy as np
from common import load_json, csv_rows, save_json


def main(args):
    root=Path(args.run).resolve(strict=True)
    report=load_json(root/'summary.json');rows=csv_rows(root/'predictions.csv')
    text=['# 个人级基线结果','',
        '这是开发期症状/量表回归。不同目标单独报告；指标不代表疾病诊断准确率。',
        '区间为按 group 配对重采样的描述性区间，未校正交叉验证训练集重叠及多次比较。','',
        f"完整外层验证：{report['complete_outer_cv']}；探索性数据：{report['exploratory']}。",'']
    intervals={}
    for target,models in report['results'].items():
        text += [f'## {target}','',
            '| 模型 | 人数 | MAE↓ | RMSE↓ | R²↑ | 相对均值 MAE 改善 [描述性95%区间] |',
            '|---|---:|---:|---:|---:|---|']
        reference=None
        for name,scores in models.items():
            rr=[r for r in rows if r['target']==target and r['model']==name]
            keys={(r['subject_id'],r['fold']) for r in rr}
            if reference is None:reference=keys
            if keys!=reference:raise ValueError('Unequal model coverage; comparison refused')
            gain=np.array([abs(float(r['y_true'])-float(r['mean_prediction']))-
                abs(float(r['y_true'])-float(r['prediction'])) for r in rr])
            groups=np.array([r['group_id'] for r in rr]);ug=np.unique(groups);rng=np.random.default_rng(20261015)
            boot=[np.concatenate([gain[groups==g] for g in rng.choice(ug,len(ug),replace=True)]).mean() for _ in range(2000)]
            ci=np.percentile(boot,[2.5,97.5]).tolist()
            intervals[target+'/'+name]={'mae_gain_vs_mean':float(gain.mean()),'descriptive_95pct':ci}
            r2='NA' if scores['r2'] is None else f"{scores['r2']:.3f}"
            text.append(f"| {name} | {scores['n']} | {scores['mae']:.3f} | {scores['rmse']:.3f} | {r2} | {gain.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
        for label,key in [('mean','mean_baseline'),('median','median_baseline')]:
            s=next(iter(models.values()))[key];r2='NA' if s['r2'] is None else f"{s['r2']:.3f}"
            text.append(f"| {label} | {len(reference)} | {s['mae']:.3f} | {s['rmse']:.3f} | {r2} | — |")
        text.append('')
    (root/'summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    save_json(root/'descriptive_intervals.json',intervals)
    print(root/'summary.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);main(p.parse_args())
