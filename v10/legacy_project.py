"""Frozen V8 engineering entry point: check, train/refit and label-free prediction."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ENGINE = HERE/'engine'
sys.path.insert(0, str(ENGINE))
import numpy as np
from threadpoolctl import threadpool_limits
from common import load_json, save_json, save_csv, sha256, new_output, manifest
from benchmark import validate_plan, protected_inputs, save_bundle, reload_bundle
from benchmark_models import predict_bundle
from improved_models import inner_records, choose_all, fit_v8_bundle, REGIONS
from improve import read_inputs

FAMILIES = {'eeg':'regional_cbramod', 'eye':'eye_regularized', 'fusion':'late_fusion'}

def lock():
    return load_json(HERE/'config/dataset_lock.json')

def verify_engine():
    if sha256(ENGINE/'CHECKSUMS.json') != lock()['engine_checksums_sha256']:
        raise ValueError('Engine checksum index changed')
    for rel, digest in load_json(ENGINE/'CHECKSUMS.json').items():
        if sha256(ENGINE/rel) != digest:
            raise ValueError('Frozen engine changed: '+rel)

def schema_check(root, expected):
    cfg = load_json(root/'config.json')
    actual = {k:v for k,v in cfg.items() if k not in ('raw_root','processed_root')}
    if actual != expected['feature_schema']:
        raise ValueError('Feature schema differs from training')
    paths = protected_inputs(root)
    params = load_json(paths[2]/'processing_config.json')['parameters']
    if params != expected['processing_parameters']:
        raise ValueError('Signal preprocessing differs from training')
    return paths

def reserve_check(plans, snapshot, expected):
    reserved = set(expected['reserve_ids'])
    union = set().union(*(set(p['subject_ids']) for p in plans))
    if union != set(expected['development_ids']) or union & reserved:
        raise ValueError('Development/reserve membership changed')
    group = {s:r.get('group_id') or s for s,r in snapshot['subjects'].items()}
    if {group[s] for s in union} & {group[s] for s in reserved}:
        raise ValueError('Identity group crosses development/reserve boundary')
    for p in plans:
        if any(p['groups'][s] != group[s] for s in p['subject_ids']):
            raise ValueError('Plan group differs from review snapshot')

def check_dataset(root):
    expected = lock()
    verify_engine()
    schema_check(root, expected)
    if sha256(root/'manifest.json') != expected['feature_manifest_sha256']:
        raise ValueError('Training manifest changed; do not silently add subjects/labels')
    if sha256(root/'review_snapshot.json') != expected['review_snapshot_sha256']:
        raise ValueError('Review snapshot changed; create a separately reviewed protocol')
    for sid, digest in expected['feature_hashes'].items():
        if sha256(root/(sid+'.npz')) != digest:
            raise ValueError('Feature file changed: '+sid)
    paths = [HERE/'config'/r['file'] for r in expected['plans']]
    plans = []
    for p,record in zip(paths, expected['plans']):
        if sha256(p) != record['sha256']:
            raise ValueError('Frozen plan changed')
        plan = load_json(p)
        validate_plan(plan, expected['feature_manifest_sha256'])
        if not plan['exploratory']:
            raise ValueError('Unverified data cannot be promoted to formal analysis')
        plans.append(plan)
    reserve_check(plans, load_json(root/'review_snapshot.json'), expected)
    return paths, plans

def labels_for(people, ids, target):
    # No reserved subject contributes a label, target normalization, or selection score.
    y = np.asarray([people[s]['targets'][target] for s in ids], float)
    if not np.isfinite(y).all():
        raise ValueError('Missing observed target in frozen development cohort')
    return y

def check_encoder(folder):
    from assets import CBRAMOD_SHA
    from extract_cbramod import CHANNELS
    e = load_json(folder/'encoder.json')
    if (e.get('weight_sha256') != CBRAMOD_SHA or e.get('channels') != CHANNELS
            or e.get('max_windows') != 0 or e.get('fitted_on_participant_data') is not False):
        raise ValueError('Encoder weight/channel/window contract differs from frozen V8')

def final_refit(root, embeddings, plans, out):
    people = manifest(root)
    entries = []
    for task,plan in enumerate(plans):
        ids = plan['subject_ids']
        x, sources, _ = read_inputs(root, embeddings, ids, people)
        index = {s:i for i,s in enumerate(ids)}
        # Refit choice is internal CV over ALL DEVELOPMENT people, not a new test.
        selection_fold = {'inner':[{'train':f['development'], 'validation':f['test']}
                                   for f in plan['folds']]}
        for target in plan['targets']:
            y = labels_for(people, ids, target)
            records = inner_records(x, y, selection_fold, index)
            choices = choose_all(records)
            save_json(out/f'selections/task_{task}_{target}.json', {
                'purpose':'full-development final refit; selection losses are NOT test performance',
                'choices':choices, 'candidates':records, 'training_ids':ids})
            for family,name in FAMILIES.items():
                chosen = choices[name]
                bundle = fit_v8_bundle(chosen['spec'], x, y)
                rel = Path('models')/target/family
                folder = out/rel
                save_bundle(folder, bundle, x, y, {
                    'target':target, 'family':family, 'name':name, 'training_ids':ids,
                    'training_groups':sorted(set(plan['groups'].values())), 'test_ids':[],
                    'exploratory':True, 'not_a_diagnosis':True,
                    'plan_sha256':lock()['plans'][task]['sha256'], 'input_hashes':sources})
                probe = x[:3].copy()
                before = predict_bundle(bundle, probe)
                after = predict_bundle(reload_bundle(folder), probe)
                if not np.allclose(before, after, rtol=0, atol=1e-12):
                    raise RuntimeError('Final model serialization replay failed')
                np.savez_compressed(folder/'replay.npz', x=probe, prediction=before)
                entries.append({'target':target, 'family':family, 'folder':rel.as_posix(),
                    'training_ids':ids, 'spec':chosen['spec'],
                    'model_sha256':sha256(folder/'model.joblib'),
                    'metadata_sha256':sha256(folder/'metadata.json'),
                    'replay_verified':True})
                print('Final model saved:', target, family, chosen['spec']['id'], flush=True)
    save_json(out/'index.json', {'exploratory':True, 'human_confirmed':False,
        'models':entries, 'dataset_lock':lock(), 'trained_on_reserve':False,
        'selection':'Original grid, fivefold internal development CV, then full-development refit',
        'performance_source':'../cv/runs/*/summary.json; never use refit training fit as test accuracy'})

def train(args):
    root = Path(args.features).resolve(strict=True)
    paths, plans = check_dataset(root)
    embeddings = Path(args.embeddings).resolve(strict=True) if args.embeddings else None
    if embeddings:
        check_encoder(embeddings)
    protected = schema_check(root, lock())+[HERE]
    if embeddings: protected.append(embeddings)
    out = new_output(args.output, *protected)
    save_json(out/'status.json', {'state':'running', 'exploratory':True})
    try:
        cmd = [sys.executable, '-B', str(ENGINE/'build_improved.py'), '--features', str(root),
               '--plans', *map(str, paths), '--output', str(out/'cv'), '--allow-unreviewed',
               '--devices', *args.devices, '--threads', str(args.threads),
               '--batch-size', str(args.batch_size)]
        if embeddings: cmd += ['--embeddings', str(embeddings)]
        with (out/'training.log').open('x', encoding='utf-8') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        embeddings = embeddings or out/'cv/regional_embeddings'
        check_encoder(embeddings)
        final_refit(root, embeddings, plans, out/'final')
        check_dataset(root)
        rows = []
        for i in range(len(plans)):
            summary = load_json(out/f'cv/runs/task_{i}/summary.json')
            for target,results in summary['results'].items():
                for family,name in FAMILIES.items():
                    r = results[name]
                    rows.append({'target':target, 'family':family, 'n':r['n'],
                        'mae':r['mae'], 'rmse':r['rmse'], 'r2':r['r2'],
                        'mean_baseline_mae':r['mean_baseline']['mae'],
                        'median_baseline_mae':r['median_baseline']['mae'], 'exploratory':True})
        save_csv(out/'metrics.csv', rows)
        save_json(out/'status.json', {'state':'complete', 'exploratory':True,
            'outer_models_verified':90, 'final_models_verified':9,
            'source_feature_hashes_unchanged':True, 'reserved_subjects_used_for_fit':0})
        print('Complete:', out, flush=True)
    except Exception as exc:
        save_json(out/'status.json', {'state':'failed', 'error':str(exc)})
        raise

def extract_for_prediction(root, ids, output, args):
    # Label-free extraction: no synthetic labels and no invented split plan.
    import torch
    from extract_cbramod import load_encoder, CHANNELS
    from extract_regional import encode
    from assets import CBRAMOD_SHA
    out = new_output(output, *protected_inputs(root), HERE)
    torch.set_num_threads(args.threads)
    model = load_encoder().to(args.device)
    record = {'status':'running', 'feature_manifest_sha256':sha256(root/'manifest.json'),
        'weight_sha256':CBRAMOD_SHA, 'channels':CHANNELS, 'regions':REGIONS,
        'regional_dimension':2000, 'max_windows':0, 'fitted_on_participant_data':False,
        'scale':'inherited uV/100; no additional normalization', 'sources':{}}
    save_json(out/'encoder.json', record)
    for sid in ids:
        source = root/(sid+'.npz')
        before = sha256(source)
        with np.load(source, allow_pickle=False) as z: windows = z['eeg'].copy()
        r,g = encode(model, windows, args.device, args.batch_size)
        dest = out/(sid+'.npz')
        np.savez_compressed(dest, regional=r, cbramod_global=g,
                            window_indices=np.arange(len(windows)))
        if sha256(source) != before: raise RuntimeError('Source changed during extraction')
        record['sources'][sid] = {'input_sha256':before, 'output_sha256':sha256(dest),
            'windows_used':len(windows), 'windows_available':len(windows)}
    record['status'] = 'complete'
    save_json(out/'encoder.json', record)
    return out

def predict(args):
    verify_engine()
    root = Path(args.features).resolve(strict=True)
    final = Path(args.models).resolve(strict=True)
    idx = load_json(final/'index.json')
    if idx['dataset_lock'] != lock(): raise ValueError('Model belongs to another protocol')
    schema_check(root, idx['dataset_lock'])
    people = manifest(root)
    ids = args.ids
    if not ids or len(ids) != len(set(ids)) or set(ids)-set(people):
        raise ValueError('Missing/duplicate/unknown prediction IDs')
    for sid in ids:
        if sid != Path(sid).name or sid in ('.','..'):
            raise ValueError('Unsafe subject ID')
        with np.load(root/(sid+'.npz'), allow_pickle=False) as z:
            if len(z['eeg']) < 3 or len(z['eye']) < 1:
                raise ValueError('Paired input quality requirement not met: '+sid)
    protected = schema_check(root, lock())+[HERE, final]
    eroot = Path(args.embeddings).resolve(strict=True) if args.embeddings else None
    if eroot: protected.append(eroot)
    out = new_output(args.output, *protected)
    eroot = eroot or extract_for_prediction(root, ids, out/'embeddings', args)
    check_encoder(eroot)
    x,hashes,_ = read_inputs(root, eroot, ids, people)
    rows = []
    for item in idx['models']:
        folder = (final/item['folder']).resolve(strict=True)
        if final not in folder.parents: raise ValueError('Model path escapes model root')
        if (sha256(folder/'model.joblib') != item['model_sha256'] or
                sha256(folder/'metadata.json') != item['metadata_sha256']):
            raise ValueError('Model or metadata changed')
        pred = predict_bundle(reload_bundle(folder), x)
        for sid,value in zip(ids,pred):
            rows.append({'subject_id':sid, 'target':item['target'], 'family':item['family'],
                'predicted_score':float(value), 'is_observed_label':False,
                'training_overlap':sid in item['training_ids'], 'exploratory':True,
                'effective_model':item['spec']['kind'], 'effective_spec':item['spec']['id']})
    for sid,h in hashes.items():
        if sha256(root/(sid+'.npz')) != h['feature']: raise RuntimeError('Source changed')
    save_csv(out/'predictions.csv', rows)
    save_json(out/'provenance.json', {'input_hashes':hashes,
        'models_index_sha256':sha256(final/'index.json'), 'observed_labels_read':False,
        'warning':'Predictions are not observed questionnaires or clinical diagnoses.'})
    print('Predictions saved:', out, flush=True)

def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('check'); p.add_argument('--features', required=True)
    p = subs.add_parser('train')
    p.add_argument('--features', required=True); p.add_argument('--output', required=True)
    p.add_argument('--embeddings'); p.add_argument('--devices', nargs='+', default=['cuda:0','cuda:1'])
    p.add_argument('--threads', type=int, default=2); p.add_argument('--batch-size', type=int, default=16)
    p = subs.add_parser('predict')
    p.add_argument('--features', required=True); p.add_argument('--models', required=True)
    p.add_argument('--ids', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--embeddings'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=2); p.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if getattr(args,'threads',1) < 1 or getattr(args,'batch_size',1) < 1:
        parser.error('threads and batch-size must be positive')
    with threadpool_limits(limits=getattr(args,'threads',2)):
        if args.command == 'check':
            _,plans = check_dataset(Path(args.features).resolve(strict=True))
            print('Verified:', [len(p['subject_ids']) for p in plans], 'exploratory cohorts')
        elif args.command == 'train': train(args)
        else: predict(args)

if __name__ == '__main__': cli()
