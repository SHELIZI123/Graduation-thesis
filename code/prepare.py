from pathlib import Path
import argparse, collections
import numpy as np
from common import read,write,sha,new_dir,csvwrite,MODALITIES
from targets import SCHEMA,parse
from heart import extract,HEART_NAMES,HEART_DIM
from core_extract import eeg_descriptors,eye_features

def main(a):
    source=Path(a.dataset); manifest=read(source/'manifest.json')
    out=new_dir(a.output,[source,Path(__file__).parent]); (out/'subjects').mkdir()
    cache=Path(a.cache_core) if a.cache_core else None
    cm=read(cache/'core.json') if cache else None
    entries=[]; targets=[]; audit=[]; issues=[]; sources=[]
    for row in manifest['subjects']:
        if a.age_policy=='exclude_known_minors' and row['age_known_minor']:continue
        if a.age_policy=='known_adults' and not row['age_known_adult']:continue
        if not any(row['modality_available'][m] for m in ['eeg','eye','ppg','nni']):continue
        sid=row['subject_id']; p=source/row['file']; sp=source/'metadata/scales'/(sid+'.json')
        h,sh=sha(p),sha(sp); values,evidence,warnings=parse(read(sp))
        # Data without labels is retained for inference but cannot enter supervised training.
        with np.load(p,allow_pickle=False) as z:
            cached=None
            if cm and sid in cm['subjects'] and cm['subjects'][sid]['current_signal_sha256']==h:
                cp=cache/cm['subjects'][sid]['file']
                if sha(cp)!=cm['subjects'][sid]['sha256']:raise ValueError('Changed cached input')
                with np.load(cp,allow_pickle=False) as cz:cached={m:cz[m+'_bag'].copy() for m in ['eeg','eye']}
            if cached is not None:eeg,eye=cached['eeg'],cached['eye']
            else:
                eeg=eeg_descriptors(z['eeg'],z['eeg_power_windows'])[-1].astype(np.float32) if row['modality_available']['eeg'] else np.empty((0,115),np.float32)
                eye=eye_features(z['eye_legacy'],z['eye_groups'])[1].astype(np.float32) if row['modality_available']['eye'] else np.empty((0,50),np.float32)
            heart,heart_info=extract(z)
        if sha(p)!=h or sha(sp)!=sh:raise ValueError('Source changed during extraction')
        if not (len(eeg) or len(eye) or len(heart)):continue
        dest=out/'subjects'/(sid+'.npz'); np.savez_compressed(dest,eeg=eeg,eye=eye,heart=heart)
        entries.append({'subject_id':sid,'group_id':row['group_id'],'file':'subjects/'+sid+'.npz','sha256':sha(dest),
            'modality_available':dict(zip(MODALITIES,map(bool,[len(eeg),len(eye),len(heart)]))),
            'instances':dict(zip(MODALITIES,map(len,[eeg,eye,heart]))),'heart_qc':heart_info,
            'historical_scope':row['historical_scope'],'age_known_adult':row['age_known_adult'],
            'age_known_minor':row['age_known_minor'],'age_missing':row['age_missing'],
            'human_confirmed':False,'visit_confirmed':False})
        targets.append([np.nan if v is None else v for v in values])
        audit.extend({'subject_id':sid,**v} for v in evidence)
        issues.extend({'subject_id':sid,**v} for v in warnings)
        sources.append({'subject_id':sid,'signal_file':row['file'],'signal_sha256':h,'scale_file':'metadata/scales/'+sid+'.json','scale_sha256':sh})
        if len(entries)%30==0:print('Prepared',len(entries),flush=True)
    y=np.asarray(targets,float); np.savez_compressed(out/'targets.npz',values=y)
    trainable=np.isfinite(y).any(1)
    summary={'subjects_with_any_signal':len(entries),'subjects_with_any_label':int(trainable.sum()),
        'targets':{t['name']:int(np.isfinite(y[:,j]).sum()) for j,t in enumerate(SCHEMA)},
        'modalities':{m:sum(e['modality_available'][m] for e in entries) for m in MODALITIES},
        'complete_three_modality_with_any_label':sum(all(e['modality_available'].values()) and bool(v) for e,v in zip(entries,trainable)),
        'labelled_subject_modality_patterns':dict(collections.Counter(''.join(str(int(e['modality_available'][m])) for m in MODALITIES) for e,v in zip(entries,trainable) if v)),
        'labelled_known_adult':sum(e['age_known_adult'] and bool(v) for e,v in zip(entries,trainable)),
        'labelled_age_unknown':sum(e['age_missing'] and bool(v) for e,v in zip(entries,trainable))}
    write(out/'manifest.json',{'state':'complete','version':'V17-three-signal-three-questionnaire','subjects':entries,
          'schema':SCHEMA,'dimensions':{'eeg':115,'eye':50,'heart':HEART_DIM},'heart_feature_names':HEART_NAMES,
          'targets_sha256':sha(out/'targets.npz'),'source_manifest_sha256':sha(source/'manifest.json'),
          'age_policy':a.age_policy,'summary':summary,'source_files':sources,
          'heart_time_unit_confirmed':False,'native_interval_values_not_converted':True,
          'physiological_HRV_in_ms_or_Hz_used':False,'modalities_synchronized':False,
          'labels_are_inputs':False,'imputed_labels':False,'exploratory':True,
          'feature_code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
    csvwrite(out/'label_provenance.csv',audit); write(out/'label_issues.json',issues)
    write(out/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache-core');p.add_argument('--age-policy',choices=['known_adults','exclude_known_minors','all_ages'],default='exclude_known_minors')
    main(p.parse_args())
