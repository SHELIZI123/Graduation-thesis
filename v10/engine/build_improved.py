"""Create a separate V8 run, keep frozen participant splits, verify every saved model."""
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import new_output,load_json,save_json
from benchmark import protected_inputs
from build_baseline import run
from assets import checked_weight


def main(args):
    checked_weight('cbramod')
    if not args.devices:raise ValueError('At least one extraction device required')
    if args.plans and not args.features:raise ValueError('--plans requires existing --features')
    protected=protected_inputs(Path(args.features).resolve(strict=True)) if args.features else [Path(args.raw_root).resolve(strict=True)]
    if args.plans:protected += [Path(p).resolve(strict=True).parent for p in args.plans]
    if args.embeddings:protected.append(Path(args.embeddings).resolve(strict=True))
    out=new_output(args.output,*protected);save_json(out/'status.json',{'state':'running','complete':False})
    try:
        if args.features:features=Path(args.features).resolve(strict=True)
        else:
            run('prepare.py',['--source',args.raw_root,'--output',out/'processed'],out/'prepare.log')
            run('features.py',['--processed',out/'processed','--output',out/'features'],out/'features.log')
            features=out/'features'
        plans=[]
        if args.plans:
            for i,p in enumerate(args.plans):
                dest=out/'plans'/f'task_{i}'/'plan.json';dest.parent.mkdir(parents=True)
                shutil.copy2(p,dest);plans.append(dest)
        else:
            for name,targets in [('emotion','depression,anxiety'),('psqi','psqi')]:
                command=['--features',features,'--output',out/'plans'/name,'--targets',targets,'--cohort','paired']
                if args.allow_unreviewed:command+=['--allow-unreviewed']
                run('make_splits.py',command,out/(name+'_plan.log'));plans.append(out/'plans'/name/'plan.json')
        if any(load_json(p)['exploratory'] for p in plans) and not args.allow_unreviewed:
            raise ValueError('Pending visit/task review: add --allow-unreviewed for exploratory analysis')
        embeddings=Path(args.embeddings).resolve(strict=True) if args.embeddings else out/'regional_embeddings'
        if not args.embeddings:
            run('extract_regional.py',['--features',features,'--plans',*plans,'--output',embeddings,
                '--device',args.devices[0],'--batch-size',args.batch_size,'--threads',args.threads,
                '--max-windows',args.max_windows],out/'extract.log')
        def worker(i,plan):
            command=['--features',features,'--plan',plan,'--embeddings',embeddings,
                '--output',out/'runs'/f'task_{i}','--threads',args.threads]
            if args.allow_unreviewed:command+=['--allow-unreviewed']
            run('improve.py',command,out/f'task_{i}.log')
        # GPU is used by the encoder; the regularized heads are small CPU regressors.
        with ThreadPoolExecutor(max_workers=min(len(args.devices),len(plans))) as pool:
            pending=[pool.submit(worker,i,p) for i,p in enumerate(plans)]
            for future in pending:future.result()
        for i,plan in enumerate(plans):
            run('verify_baseline.py',['--run',out/'runs'/f'task_{i}','--plan',plan,'--threads',args.threads],out/f'verify_{i}.log')
            run('compare_baselines.py',['--run',out/'runs'/f'task_{i}'],out/f'report_{i}.log')
        save_json(out/'status.json',{'state':'complete','complete':True,'features':str(features),
            'embeddings':str(embeddings),'plans':[str(p) for p in plans],
            'exploratory':any(load_json(p)['exploratory'] for p in plans)})
    except Exception as exc:
        save_json(out/'status.json',{'state':'failed','complete':False,'error':str(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--features');group.add_argument('--raw-root')
    p.add_argument('--plans',nargs='+');p.add_argument('--output',required=True);p.add_argument('--embeddings')
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1'])
    p.add_argument('--threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-windows',type=int,default=0);p.add_argument('--allow-unreviewed',action='store_true')
    main(p.parse_args())
