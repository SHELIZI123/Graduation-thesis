"""Prepare data or reuse V3 features, freeze plans, extract CBraMod and run two independent jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
from common import load_json, save_json, new_output
from benchmark import protected_inputs
from benchmark_models import DEFAULT_MODELS, SPECS
from assets import checked_weight

CODE=Path(__file__).resolve().parent


def run(script,args,log):
    command=[sys.executable,'-B',str(CODE/script),*map(str,args)]
    print('Running',script,'log:',log,flush=True)
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1')
    with Path(log).open('x',encoding='utf-8') as file:
        p=subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,cwd=CODE,env=env)
    if p.returncode:raise RuntimeError(f'{script} failed ({p.returncode}); see {log}')


def main(args):
    checked_weight('cbramod')
    if args.with_tabpfn:checked_weight('tabpfn')
    if not args.devices:raise ValueError('Specify at least one device')
    if args.plans and not args.features:raise ValueError('--plans requires --features')
    protected=protected_inputs(Path(args.features).resolve(strict=True)) if args.features else [Path(args.raw_root).resolve(strict=True)]
    if args.plans:protected += [Path(p).resolve(strict=True).parent for p in args.plans]
    if args.embeddings:protected.append(Path(args.embeddings).resolve(strict=True))
    out=new_output(args.output,*protected)
    save_json(out/'status.json',{'state':'running','complete':False})
    try:
        if args.features:
            features=Path(args.features).resolve(strict=True)
        else:
            run('prepare.py',['--source',args.raw_root,'--output',out/'processed'],out/'prepare.log')
            run('features.py',['--processed',out/'processed','--output',out/'features'],out/'features.log')
            features=out/'features'
        plans=[Path(p).resolve(strict=True) for p in args.plans] if args.plans else []
        if not plans:
            for task,targets in [('emotion','depression,anxiety'),('psqi','psqi')]:
                cmd=['--features',features,'--output',out/'plans'/task,'--targets',targets,'--cohort','paired']
                if args.allow_unreviewed:cmd+=['--allow-unreviewed']
                if args.reserve_ids:cmd+=['--reserve-ids',args.reserve_ids]
                run('make_splits.py',cmd,out/(task+'_plan.log'))
                plans.append(out/'plans'/task/'plan.json')
        if any(load_json(p)['exploratory'] for p in plans) and not args.allow_unreviewed:
            raise ValueError('Exploratory plans supplied: add --allow-unreviewed explicitly')
        embeddings=Path(args.embeddings).resolve(strict=True) if args.embeddings else out/'embeddings'
        if not args.embeddings:
            run('extract_cbramod.py',['--features',features,'--output',embeddings,'--plans',*plans,
                '--device',args.devices[0],'--batch-size',args.batch_size,'--threads',args.threads,
                '--max-windows',args.max_windows],out/'extract.log')
        models=list(DEFAULT_MODELS)+(['eye_tabpfn'] if args.with_tabpfn else [])
        def worker(device,indexed):
            for i,plan in indexed:
                run('benchmark.py',['--features',features,'--plan',plan,'--embeddings',embeddings,
                    '--models',','.join(models),'--output',out/'runs'/('task_'+str(i)),
                    '--device',device,'--threads',args.threads],out/f'task_{i}.log')
        # A separate process per task; no DDP and no synthetic window alignment.
        with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
            pending=[pool.submit(worker,device,[(i,p) for i,p in enumerate(plans) if i%len(args.devices)==j])
                for j,device in enumerate(args.devices)]
            errors=[]
            for future in pending:
                try:future.result()
                except Exception as exc:errors.append(str(exc))
            if errors:raise RuntimeError('; '.join(errors))
        for i,plan in enumerate(plans):
            run('verify_baseline.py',['--run',out/'runs'/('task_'+str(i)),'--plan',plan,
                '--device',args.devices[i%len(args.devices)],'--threads',args.threads],out/f'verify_{i}.log')
            run('compare_baselines.py',['--run',out/'runs'/('task_'+str(i))],out/f'report_{i}.log')
        save_json(out/'status.json',{'state':'complete','complete':True,'features':str(features),
            'plans':[str(p) for p in plans],'models':models,'exploratory':any(load_json(p)['exploratory'] for p in plans)})
    except Exception as exc:
        save_json(out/'status.json',{'state':'failed','complete':False,'error':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--raw-root');source.add_argument('--features')
    p.add_argument('--plans',nargs='+');p.add_argument('--output',required=True)
    p.add_argument('--embeddings',help='Reuse verified frozen extraction; per-ID source hashes are checked')
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1'])
    p.add_argument('--with-tabpfn',action='store_true');p.add_argument('--allow-unreviewed',action='store_true')
    p.add_argument('--reserve-ids');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-windows',type=int,default=0);p.add_argument('--threads',type=int,default=2)
    main(p.parse_args())
