"""Copy portable feature files and edit only the NEW copy's path configuration."""
import argparse
import shutil
from pathlib import Path
from common import load_json,save_json,sha256,new_output


def main(args):
    source=Path(args.source).resolve(strict=True);cfg=load_json(source/'config.json')
    raw=Path(args.raw_root).resolve(strict=True);processed=Path(args.processed_root).resolve(strict=True)
    out=new_output(args.output,source,raw,processed)
    files=[source/n for n in ['manifest.json','review_snapshot.json']]
    files += sorted(source.glob('*.npz'))
    sources={}
    for p in files:
        before=sha256(p);dest=out/p.name;shutil.copy2(p,dest)
        if sha256(dest)!=before or sha256(p)!=before:raise RuntimeError('Copy verification failed')
        sources[p.name]=before
    cfg.update(raw_root=str(raw),processed_root=str(processed));save_json(out/'config.json',cfg)
    save_json(out/'relocation.json',{'source':str(source),'source_config_sha256':sha256(source/'config.json'),
        'files_sha256':sources,'note':'Only this new copy of config.json was relocated; no signal values or manifest changed'})
    print('Copied',len(files),'files; manifest SHA',sha256(out/'manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--raw-root',required=True);p.add_argument('--processed-root',required=True)
    main(p.parse_args())
