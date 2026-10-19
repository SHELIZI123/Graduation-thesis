"""Per-record features; no cohort-fitted transforms and no synthetic labels."""
import argparse
from pathlib import Path
import numpy as np
from scipy.signal import welch
from common import manifest, load_json, save_json, new_output, csv_rows, yes, sha256

BANDS = [(1,4), (4,8), (8,13), (13,30), (30,45)]


def eeg_features(x):
    # x is already uV/100. Do not divide again.
    f, psd = welch(x, fs=200, nperseg=400, axis=-1)
    power = np.stack([np.sum(psd[..., (f>=a)&(f<b)], axis=-1)*(f[1]-f[0]) for a,b in BANDS], -1)
    relative = power / np.maximum(power.sum(-1, keepdims=True), 1e-12)
    spectral = np.concatenate([np.log(np.maximum(power, 1e-12)).reshape(len(x), -1),
                               relative.reshape(len(x), -1)], axis=1)
    centered = x - x.mean(-1, keepdims=True)
    cov = centered @ centered.transpose(0,2,1) / max(x.shape[-1]-1, 1)
    scale = np.trace(cov, axis1=1, axis2=2)/x.shape[1]
    cov = .9*cov + .1*scale[:,None,None]*np.eye(x.shape[1])[None]
    val, vec = np.linalg.eigh(cov)
    logcov = (vec * np.log(np.maximum(val, 1e-10))[:,None,:]) @ vec.transpose(0,2,1)
    a,b = np.triu_indices(x.shape[1])
    tangent = logcov[:,a,b] * np.where(a==b, 1., np.sqrt(2.))
    return spectral.astype('float32'), tangent.astype('float32')


def eye_stats(x):
    delta = np.diff(x, axis=-1)*62.5
    return np.concatenate([x.mean(-1), x.std(-1), np.quantile(x,.75,axis=-1)-np.quantile(x,.25,axis=-1),
                           np.abs(delta).mean(-1), delta.std(-1)], axis=1).astype('float32')


def balanced_stats(features, groups):
    """Equal weight per measurement, regardless of its duration/window count."""
    a = np.array([np.median(features[groups==g], axis=0) for g in np.unique(groups)])
    return np.r_[a.mean(0), a.std(0)].astype('float32')


def target_features(native, start, end):
    t = native['time_s']; x = native['channel1_xyz'][:,:2]; s = native['stimulus_xyz'][:,:2]
    valid = (t>=start)&(t<end)&native['channel1_valid']&np.isfinite(x).all(1)&np.isfinite(s).all(1)
    if valid.sum()<20:
        return np.full(6,np.nan,'float32')
    x,s=x[valid],s[valid]
    err=x-s
    corr=[np.corrcoef(x[:,i],s[:,i])[0,1] if x[:,i].std()>1e-8 and s[:,i].std()>1e-8 else 0. for i in range(2)]
    return np.r_[np.sqrt((err**2).mean(0)), err.mean(0), corr].astype('float32')


def nni_features(records, review):
    unit=review.get('nni_unit','').strip().lower()
    if unit not in {'seconds','milliseconds'}:
        return np.full(8,np.nan,'float32')
    if len(records)==1:
        record=records[0]
    else:
        chosen=[r for r in records if str(r['record_sequence'])==review.get('hrv_record_sequence','')]
        if len(chosen)!=1:
            return np.full(8,np.nan,'float32')
        record=chosen[0]
    rr=np.asarray(record['nni'],float)*(1000 if unit=='seconds' else 1)
    # No deletion-and-joining of bad beats: reject the record for manual review.
    if len(rr)<100 or not np.isfinite(rr).all() or np.any((rr<300)|(rr>2000)) or rr.sum()<120000:
        return np.full(8,np.nan,'float32')
    diff=np.diff(rr)
    return np.array([60000/rr.mean(),rr.mean(),rr.std(ddof=1),np.sqrt(np.mean(diff**2)),
                     np.mean(np.abs(diff)>50),np.quantile(rr,.75)-np.quantile(rr,.25),rr.min(),rr.max()], 'float32')


def main(args):
    source=Path(args.processed).resolve(strict=True)
    out=new_output(args.output,source,load_json(source/'processing_config.json')['source_root'])
    people=manifest(source)
    reviews={r['subject_id']:r for r in csv_rows(args.subject_review or source/'review/subjects.csv')}
    taskrows=csv_rows(args.task_review or source/'review/eye_tasks.csv')
    tasks={(r['subject_id'],r['measurement_id']):r for r in taskrows}
    save_json(out/'review_snapshot.json',{'subjects':reviews,'eye_tasks':taskrows})
    event_filter=load_json(args.eeg_event_filter) if args.eeg_event_filter else None
    if event_filter is not None and (not isinstance(event_filter,dict) or not event_filter):
        raise ValueError('Event filter must be a nonempty mapping of exact EDF label to digital code')
    included=[]
    for sid,row in people.items():
        folder=source/'subjects'/sid
        e=np.empty((0,16,800),'float32'); ep=np.empty((0,160),'float32'); ec=np.empty((0,136),'float32')
        if row.get('eeg_windows',0):
            e=np.load(folder/'eeg_windows.npy',allow_pickle=False).reshape(-1,16,800)
            if event_filter:
                metadata=csv_rows(folder/'eeg_window_metadata.csv')
                import json
                states=[json.loads(r['event_states_json']) for r in metadata]
                keep=[i for i,s in enumerate(states) if all(k in s and s[k]==v for k,v in event_filter.items())]
                e=e[keep]
            if len(e):ep,ec=eeg_features(e)
        x=np.empty((0,2,250),'float32'); eye=np.empty((0,4,250),'float32')
        ef=np.empty((0,10),'float32'); et=np.empty((0,6),'float32'); groups=np.empty(0,'int64')
        if row.get('eye_windows',0):
            x=np.load(folder/'eye_windows.npy',allow_pickle=False).reshape(-1,2,250)
            eye=np.concatenate([x,np.diff(x,axis=-1,prepend=x[...,:1])*62.5],axis=1)
            ef=eye_stats(x)
            meta=csv_rows(folder/'eye_window_metadata.csv')
            names=sorted({r['measurement_id'] for r in meta})
            groups=np.array([names.index(r['measurement_id']) for r in meta],dtype='int64')
            et=np.full((len(meta),6),np.nan,'float32')
            native={}
            for i,r in enumerate(meta):
                mid=r['measurement_id']; review=tasks.get((sid,mid),{})
                if yes(review.get('coordinate_frame_confirmed')) and yes(review.get('task_confirmed')):
                    if mid not in native:
                        with np.load(folder/'eye_native'/(mid+'.npz'),allow_pickle=False) as z:
                            native[mid]={k:z[k] for k in z.files}
                    et[i]=target_features(native[mid],float(r['start_s']),float(r['end_s']))
        base=balanced_stats(ef,groups) if len(ef) else np.full(20,np.nan,'float32')
        good=np.isfinite(et).all(1)
        relative=balanced_stats(et[good],groups[good]) if good.any() else np.full(12,np.nan,'float32')
        hr=nni_features(load_json(folder/'nni_records.json'),reviews.get(sid,{}))
        np.savez_compressed(out/(sid+'.npz'),eeg=e,eeg_power_windows=ep,eye=eye,
            eye_feature_windows=ef,eye_target_windows=np.nan_to_num(et),eye_target_mask=good,
            eye_groups=groups,eeg_power=np.median(ep,0) if len(ep) else np.full(160,np.nan),
            eeg_cov=np.median(ec,0) if len(ec) else np.full(136,np.nan),eye_base=base,eye_target=relative,hrv=hr)
        included.append({**row,'eeg_windows':len(e),'file':sid+'.npz','hrv_usable':bool(np.isfinite(hr).all()),
                         'eye_target_usable':bool(good.any())})
    save_json(out/'manifest.json',{'format_version':3,'subjects':included})
    save_json(out/'config.json',{'processed_root':str(source),'raw_root':load_json(source/'processing_config.json')['source_root'],
        'spectral_bands_hz':BANDS,'covariance':'0.1 isotropic shrinkage; matrix log at identity; upper triangle; NOT learned Riemannian reference',
        'eye_base':'10 window statistics; median within measurement; mean/std across measurements',
        'eye_target':'RMS error xy, mean error xy, correlation xy; only manually confirmed task AND coordinate frame',
        'hrv':'NNI record units and visit selection from review; time-domain only; PPG-derived PRV',
        'eeg_event_filter':event_filter,'event_semantics':'provided by user; no automatic physiological naming',
        'no_cohort_fitted_preprocessing':True})
    print('Feature subjects:',len(included),'target-eye:',sum(r['eye_target_usable'] for r in included),
          'HRV:',sum(r['hrv_usable'] for r in included),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed',required=True);p.add_argument('--output',required=True)
    p.add_argument('--subject-review');p.add_argument('--task-review')
    p.add_argument('--eeg-event-filter',help='JSON of exact event labels/digital values; use only after protocol verification')
    main(p.parse_args())
