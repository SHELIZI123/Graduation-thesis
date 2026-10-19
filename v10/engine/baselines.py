"""Nested grouped CV for strong small-data baselines, with common held-out IDs."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from common import manifest,load_json,save_json,save_csv,new_output,metrics,validate_folds,sha256

FEATURE_SETS={'eeg_power':['eeg_power'],'eeg_cov':['eeg_cov'],'eye':['eye_base'],
    'eye_target':['eye_base','eye_target'],'fusion':['eeg_power','eeg_cov','eye_base'],
    'fusion_target':['eeg_power','eeg_cov','eye_base','eye_target'],'hrv':['hrv'],
    'triple':['eeg_power','eeg_cov','eye_base','hrv']}


class SafeImputer(BaseEstimator,TransformerMixin):
    """Keep even all-missing columns; fit medians exclusively on training rows."""
    def fit(self,x,y=None):
        x=np.asarray(x,float)
        self.fill_=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
        return self
    def transform(self,x):
        x=np.asarray(x,float)
        return np.where(np.isfinite(x),x,self.fill_)


def read_features(root,ids,feature_set,people):
    result=[]
    for sid in ids:
        if feature_set=='demographics':
            r=people[sid];x=[r.get('age') or np.nan,{'男':1,'女':0}.get(r.get('sex'),np.nan)]
        else:
            with np.load(Path(root)/(sid+'.npz'),allow_pickle=False) as z:
                x=np.concatenate([z[k] for k in FEATURE_SETS[feature_set]])
                for k in FEATURE_SETS[feature_set]:
                    if not np.isfinite(z[k]).any():
                        raise ValueError(f'{sid}: {k} unavailable; freeze an appropriate common cohort first')
        result.append(x)
    return np.asarray(result,float)


def estimator(config):
    model=Ridge(alpha=config['alpha']) if config['model']=='ridge' else MultiOutputRegressor(SVR(C=config['C'],epsilon=.1,gamma='scale'))
    return make_pipeline(SafeImputer(),StandardScaler(),model)


def main(args):
    root=Path(args.features).resolve(strict=True);plan=load_json(args.plan);validate_folds(plan)
    if sha256(root/'manifest.json')!=plan['feature_manifest_sha256']:raise ValueError('Feature manifest changed')
    cfg=load_json(root/'config.json')
    out=new_output(args.output,root,cfg['raw_root'],cfg['processed_root'])
    people=manifest(root);ids=plan['subject_ids'];idx={s:i for i,s in enumerate(ids)};targets=plan['targets']
    x=read_features(root,ids,args.feature_set,people)
    y=np.array([[people[s]['targets'][t] for t in targets] for s in ids],float)
    candidates=[{'model':'ridge','alpha':a} for a in [.1,1.,10.,100.,1000.]]
    candidates += [{'model':'svr','C':c} for c in [.1,1.,10.]]
    all_rows=[]
    for fold in plan['folds']:
        if args.fold is not None and fold['fold']!=args.fold:continue
        folder=out/f"fold_{fold['fold']}";folder.mkdir()
        scores=[]
        for config in candidates:
            losses=[]
            for inner in fold['inner']:
                a=[idx[s] for s in inner['train']];b=[idx[s] for s in inner['validation']]
                mean=y[a].mean(0);std=np.maximum(y[a].std(0),1e-6)
                model=estimator(config);model.fit(x[a],(y[a]-mean)/std)
                pred=np.asarray(model.predict(x[b])).reshape(len(b),-1)*std+mean
                losses.append(float(np.mean(np.abs(pred-y[b])/std)))
            scores.append({'config':config,'inner_standardized_mae':losses,'mean':float(np.mean(losses))})
        selected=min(scores,key=lambda r:r['mean'])['config']
        # Persist decision before reading outer-test targets into evaluation.
        save_json(folder/'selection.json',{'selected':selected,'candidates':scores,
                     'training_ids':fold['development'],'test_ids':fold['test']})
        a=[idx[s] for s in fold['development']];b=[idx[s] for s in fold['test']]
        mean=y[a].mean(0);std=np.maximum(y[a].std(0),1e-6);median=np.median(y[a],0)
        model=estimator(selected);model.fit(x[a],(y[a]-mean)/std)
        pred=np.asarray(model.predict(x[b])).reshape(len(b),-1)*std+mean
        rows=[]
        for j,sid in enumerate(fold['test']):
            for k,t in enumerate(targets):
                rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'fold':fold['fold'],
                    'target':t,'y_true':float(y[b[j],k]),'prediction':float(pred[j,k]),
                    'mean_prediction':float(mean[k]),'median_prediction':float(median[k])})
        save_csv(folder/'predictions.csv',rows);all_rows+=rows
        save_json(folder/'normalization.json',{'fit_ids':fold['development'],'target_mean':mean.tolist(),
                    'target_std':std.tolist(),'feature_mean':model[1].mean_.tolist(),
                    'feature_scale':model[1].scale_.tolist(),'imputer':model[0].fill_.tolist()})
        print('Completed fold',fold['fold'],selected,flush=True)
    save_csv(out/'predictions.csv',all_rows)
    report={}
    for t in targets:
        rr=[r for r in all_rows if r['target']==t];truth=[r['y_true'] for r in rr]
        report[t]={key:metrics(truth,[r[key] for r in rr]) for key in ['prediction','mean_prediction','median_prediction']}
        report[t]['n']=len(rr)
    save_json(out/'summary.json',{'results':report,'feature_set':args.feature_set,'plan_sha256':sha256(args.plan),
                 'exploratory':plan['exploratory'],'complete_outer_cv':args.fold is None,
                 'parameter_selection':'nested inner CV','raw_outputs_not_clipped':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--feature-set',choices=list(FEATURE_SETS)+['demographics'],required=True)
    p.add_argument('--fold',type=int)
    main(p.parse_args())
