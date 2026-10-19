"""Nested subject/group CV for lightweight neural regressors (one GPU per job)."""
import argparse
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from models import SubjectModel
from common import manifest,load_json,save_json,save_csv,new_output,metrics,validate_folds,sha256


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def fit_norm(root,ids,people,targets):
    result={'fit_ids':ids}
    for key in ['eeg','eye','eeg_power_windows','eye_feature_windows']:
        means=[];seconds=[]
        for sid in ids:
            with np.load(root/(sid+'.npz'),allow_pickle=False) as z:
                x=z[key].astype('float64')
            if not len(x):continue
            if x.ndim==3:x=x.transpose(0,2,1).reshape(-1,x.shape[1])
            means.append(x.mean(0));seconds.append((x*x).mean(0))
        if means:
            mean=np.mean(means,0);std=np.sqrt(np.maximum(np.mean(seconds,0)-mean*mean,1e-8))
        else:
            dim={'eeg':16,'eye':4,'eeg_power_windows':160,'eye_feature_windows':10}[key]
            mean=np.zeros(dim);std=np.ones(dim)
        result[key]={'mean':mean.tolist(),'std':std.tolist()}
    y=np.array([[people[s]['targets'][t] for t in targets] for s in ids],float)
    result.update(target_mean=y.mean(0).tolist(),target_std=np.maximum(y.std(0),1e-6).tolist(),target_median=np.median(y,0).tolist())
    return result


class Bags(Dataset):
    def __init__(self,root,ids,people,targets,norm,training=False,seed=0,views=8,measurements=24,eye_views=4):
        self.root=root;self.ids=ids;self.people=people;self.targets=targets;self.norm=norm
        self.training=training;self.seed=seed;self.views=views;self.measurements=measurements;self.eye_views=eye_views
        self.epoch=0;self.cache={}
    def __len__(self):return len(self.ids)
    def normalized(self,x,key):
        m=np.array(self.norm[key]['mean'],'float32');s=np.array(self.norm[key]['std'],'float32')
        if x.ndim==3:m=m[None,:,None];s=s[None,:,None]
        return (x-m)/s
    def __getitem__(self,index):
        sid=self.ids[index]
        if sid not in self.cache:
            with np.load(self.root/(sid+'.npz'),allow_pickle=False) as z:
                self.cache[sid]={k:z[k] for k in ['eeg','eye','eeg_power_windows','eye_feature_windows','eye_groups']}
        z=self.cache[sid]
        re=np.random.default_rng(np.random.SeedSequence([self.seed,int(sid),self.epoch if self.training else 0,1]))
        rg=np.random.default_rng(np.random.SeedSequence([self.seed,int(sid),self.epoch if self.training else 0,2]))
        def choose(n,k,rng):
            return rng.choice(n,k,replace=n<k) if self.training else np.linspace(0,n-1,k).round().astype(int)
        if len(z['eeg']):
            ei=choose(len(z['eeg']),self.views,re)
            eeg=self.normalized(z['eeg'][ei],'eeg');power=self.normalized(z['eeg_power_windows'][ei],'eeg_power_windows')
        else:eeg=np.zeros((self.views,16,800),'float32');power=np.zeros((self.views,160),'float32')
        groups=np.unique(z['eye_groups'])
        if len(groups)>self.measurements:
            groups=groups[choose(len(groups),self.measurements,rg)]
        eye=np.zeros((self.measurements,self.eye_views,4,250),'float32')
        stats=np.zeros((self.measurements,self.eye_views,10),'float32');mask=np.zeros(self.measurements,bool)
        for i,g in enumerate(groups):
            pool=np.flatnonzero(z['eye_groups']==g);selected=pool[choose(len(pool),self.eye_views,rg)]
            eye[i]=self.normalized(z['eye'][selected],'eye');stats[i]=self.normalized(z['eye_feature_windows'][selected],'eye_feature_windows');mask[i]=True
        target=np.array([self.people[sid]['targets'][t] for t in self.targets],'float32')
        return {'eeg':eeg,'power':power,'eye':eye,'eye_stats':stats,'measurement_mask':mask,
                'y':((target-self.norm['target_mean'])/self.norm['target_std']).astype('float32')}


def loader(dataset,batch,shuffle=False,seed=0):
    g=torch.Generator();g.manual_seed(seed)
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=0,generator=g)


def device_batch(batch,device):return {k:v.to(device) for k,v in batch.items()}


@torch.no_grad()
def predict(model,dataset,device,batch_size):
    model.eval();values=[]
    for batch in loader(dataset,batch_size):
        values.append(model(device_batch(batch,device)).cpu().numpy())
    return np.concatenate(values)


def train_once(args,root,people,targets,train_ids,val_ids,lr,seed,max_epochs,fixed=False):
    set_seed(seed);device=torch.device(args.device)
    norm=fit_norm(root,train_ids,people,targets)
    train=Bags(root,train_ids,people,targets,norm,True,seed,args.views,args.measurements,args.eye_views)
    val=Bags(root,val_ids,people,targets,norm,False,seed,args.eval_views,args.measurements,args.eye_views) if val_ids else None
    model=SubjectModel(args.modality,len(targets),args.pooling,args.fusion,args.dropout).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    truth=np.array([[people[s]['targets'][t] for t in targets] for s in val_ids]) if val_ids else None
    if val is not None:
        truth=(truth-norm['target_mean'])/norm['target_std']
        best=float(np.abs(truth).mean())
    else:best=None
    best_epoch=0;state=copy.deepcopy(model.state_dict());history=[];wait=0
    data_loader=loader(train,args.batch_size,True,seed)
    for epoch in range(1,max_epochs+1):
        train.epoch=epoch;model.train();loss_sum=0.;count=0
        for batch in data_loader:
            batch=device_batch(batch,device);optimizer.zero_grad(set_to_none=True)
            prediction=model(batch)
            loss=torch.nn.functional.mse_loss(prediction,batch['y'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradients')
            optimizer.step();loss_sum+=float(loss.detach())*len(prediction);count+=len(prediction)
        record={'epoch':epoch,'train_normalized_mse':loss_sum/count}
        if val is not None:
            p=predict(model,val,device,args.batch_size);score=float(np.abs(p-truth).mean())
            record.update(validation_normalized_mae=score,prediction_std=p.std(0).tolist())
            if score<best-1e-6:
                best=score;best_epoch=epoch;state=copy.deepcopy(model.state_dict());wait=0
            else:wait+=1
        else:state=copy.deepcopy(model.state_dict());best_epoch=epoch
        history.append(record)
        if val is not None and wait>=args.patience:break
    model.load_state_dict(state)
    return model,norm,history,best,best_epoch


def main(args):
    torch.set_num_threads(args.cpu_threads)
    root=Path(args.features).resolve(strict=True);plan=load_json(args.plan);validate_folds(plan)
    if sha256(root/'manifest.json')!=plan['feature_manifest_sha256']:raise ValueError('Feature manifest changed')
    cfg=load_json(root/'config.json');out=new_output(args.output,root,cfg['raw_root'],cfg['processed_root'])
    people=manifest(root);targets=plan['targets'];all_rows=[]
    for sid in plan['subject_ids']:
        row=people[sid]
        if args.modality in {'eeg','fusion'} and row['eeg_windows']<1:raise ValueError('Missing EEG input')
        if args.modality in {'eye','fusion'} and row['eye_windows']<1:raise ValueError('Missing eye input')
    for fold in plan['folds']:
        if args.fold is not None and args.fold!=fold['fold']:continue
        folder=out/f"fold_{fold['fold']}";folder.mkdir()
        candidates=[]
        for lr in args.learning_rates:
            scores=[];epochs=[]
            for j,inner in enumerate(fold['inner']):
                seed=args.seed+100*fold['fold']+j
                model,norm,history,score,epoch=train_once(args,root,people,targets,inner['train'],inner['validation'],lr,seed,args.epochs)
                save_json(folder/f'inner_lr{lr}_{j}.json',{'history':history,'best_epoch':epoch,'best_score':score,
                          'normalization':norm,'validation_ids':inner['validation']})
                scores.append(score);epochs.append(epoch);del model
                if torch.cuda.is_available():torch.cuda.empty_cache()
            candidates.append({'lr':lr,'inner_mae':scores,'mean':float(np.mean(scores)),
                               'refit_epochs':int(np.median(epochs)),'inner_best_epochs':epochs})
        selected=min(candidates,key=lambda c:c['mean'])
        save_json(folder/'selection.json',{'selected':selected,'candidates':candidates,
                    'refit_rule':'median inner-selected epochs; can be zero (training mean)',
                    'development_ids':fold['development'],'test_ids':fold['test']})
        seed=args.seed+100*fold['fold']+99
        model,norm,history,_,_=train_once(args,root,people,targets,fold['development'],[],selected['lr'],seed,selected['refit_epochs'],True)
        test=Bags(root,fold['test'],people,targets,norm,False,seed,args.eval_views,args.measurements,args.eye_views)
        p=predict(model,test,torch.device(args.device),args.batch_size)*norm['target_std']+norm['target_mean']
        rows=[]
        for i,sid in enumerate(fold['test']):
            for k,t in enumerate(targets):
                rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'fold':fold['fold'],'target':t,
                    'y_true':people[sid]['targets'][t],'prediction':float(p[i,k]),
                    'mean_prediction':norm['target_mean'][k],'median_prediction':norm['target_median'][k]})
        save_csv(folder/'predictions.csv',rows);all_rows+=rows
        save_json(folder/'normalization.json',norm);save_json(folder/'refit_history.json',history)
        torch.save({'state_dict':model.cpu().state_dict(),'normalization':norm,'args':vars(args),
                    'targets':targets,'development_ids':fold['development']},folder/'checkpoint.pt')
        print('Completed fold',fold['fold'],selected,flush=True)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    save_csv(out/'predictions.csv',all_rows)
    report={}
    for t in targets:
        rr=[r for r in all_rows if r['target']==t];truth=[r['y_true'] for r in rr]
        report[t]={key:metrics(truth,[r[key] for r in rr]) for key in ['prediction','mean_prediction','median_prediction']}
        report[t]['n']=len(rr)
    save_json(out/'summary.json',{'results':report,'args':vars(args),'plan_sha256':sha256(args.plan),
                 'complete_outer_cv':args.fold is None,'exploratory':plan['exploratory'],
                 'evaluation':'nested inner model selection + fixed-epoch development refit + outer test'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modality',choices=['eeg','eye','fusion'],default='fusion')
    p.add_argument('--pooling',choices=['mean','attention'],default='attention')
    p.add_argument('--fusion',choices=['concat','gate'],default='gate')
    p.add_argument('--learning-rates',nargs='+',type=float,default=[.0001,.0003])
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--views',type=int,default=8);p.add_argument('--eval-views',type=int,default=32)
    p.add_argument('--measurements',type=int,default=24);p.add_argument('--eye-views',type=int,default=4)
    p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20261013);p.add_argument('--fold',type=int)
    main(p.parse_args())
