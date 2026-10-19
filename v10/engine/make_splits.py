"""Freeze common cohorts and nested folds BEFORE training models."""
import argparse
from pathlib import Path
import numpy as np
from common import manifest,load_json,save_json,new_output,yes,validate_folds,sha256


def group_partitions(ids, groups, n, seed):
    unique=sorted({groups[s] for s in ids})
    if len(unique)<n:
        raise ValueError('Not enough independent groups for requested fold count')
    rng=np.random.default_rng(seed)
    bins=np.array_split(rng.permutation(unique),n)
    return [sorted(s for s in ids if groups[s] in set(b)) for b in bins]


def main(args):
    root=Path(args.features).resolve(strict=True)
    cfg=load_json(root/'config.json')
    out=new_output(args.output,root,cfg['raw_root'],cfg['processed_root'])
    people=manifest(root);review=load_json(root/'review_snapshot.json')['subjects']
    reserved=set(Path(args.reserve_ids).read_text(encoding='utf-8-sig').split()) if args.reserve_ids else set()
    if reserved-set(people):raise ValueError('Unknown reserved IDs')
    targets=args.targets.split(','); ids=[]; exclusions={};groups={}
    if set(targets)-{'psqi','depression','anxiety'}:raise ValueError('Unsupported target')
    for sid,row in people.items():
        why=[];r=review.get(sid,{})
        groups[sid]=r.get('group_id','').strip() or sid
        if sid in reserved:why.append('reserved')
        if row.get('age') is None or row['age']<18:why.append('adult_age_unconfirmed')
        if any(row['targets'].get(t) is None for t in targets):why.append('missing_unique_target')
        if args.cohort in {'paired','triple','target_eye','eeg'} and row.get('eeg_windows',0)<3:why.append('eeg_quality')
        if args.cohort in {'paired','triple','target_eye','eye'} and row.get('eye_windows',0)<1:why.append('eye_quality')
        if args.cohort=='triple' and not row['hrv_usable']:why.append('hrv_unconfirmed_or_missing')
        if args.cohort=='target_eye' and not row['eye_target_usable']:why.append('eye_target_unconfirmed')
        if not args.allow_unreviewed:
            required=['include','identity_confirmed','visit_confirmed']
            if args.cohort!='eye':required+=['eeg_protocol_confirmed']
            if args.cohort!='eeg':required+=['eye_protocol_confirmed']
            if not all(yes(r.get(k)) for k in required):why.append('pending_review')
        if why:exclusions[sid]=why
        else:ids.append(sid)
    # A reserved person's other ID must also be excluded.
    reserved_groups={groups[s] for s in reserved}
    for sid in list(ids):
        if groups[sid] in reserved_groups:
            ids.remove(sid);exclusions[sid]=['same_person_as_reserved_id']
    if len(ids)<max(15,args.outer*3):
        save_json(out/'exclusions.json',exclusions)
        raise ValueError('Too few eligible subjects; review labels/protocols or use explicit exploratory flag')
    folds=[]
    for i,test in enumerate(group_partitions(ids,groups,args.outer,args.seed)):
        dev=sorted(set(ids)-set(test))
        inner=[{'train':sorted(set(dev)-set(v)),'validation':v}
               for v in group_partitions(dev,groups,args.inner,args.seed+100+i)]
        folds.append({'fold':i,'development':dev,'test':test,'inner':inner})
    plan={'targets':targets,'cohort':args.cohort,'subject_ids':ids,'groups':{s:groups[s] for s in ids},
          'reserved_ids':sorted(reserved),'seed':args.seed,'folds':folds,'exploratory':args.allow_unreviewed,
          'feature_manifest_sha256':sha256(root/'manifest.json'),'exclusions':exclusions,
          'selection_rule':'inner validation only; outer tests never select epoch/hyperparameters',
          'scores_used_for_partition':False}
    validate_folds(plan);save_json(out/'plan.json',plan)
    print('Frozen cohort:',len(ids),'IDs;',len({groups[s] for s in ids}),'groups;',len(folds),'outer folds')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    p.add_argument('--targets',default='depression,anxiety')
    p.add_argument('--cohort',choices=['paired','triple','target_eye','eeg','eye'],default='paired')
    p.add_argument('--outer',type=int,default=5);p.add_argument('--inner',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261013);p.add_argument('--reserve-ids')
    p.add_argument('--allow-unreviewed',action='store_true')
    main(p.parse_args())
