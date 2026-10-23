"""Import verified private runtime files; never publish or mutate the source bundle."""
from pathlib import Path,PurePosixPath
import argparse,json,shutil
from verify_public_source import ROOT,sha,verify

def relative(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe package path')
    return p

def main(a):
    source=Path(a.source).resolve();root=ROOT.resolve()
    if source==root or root in source.parents or source in root.parents:raise ValueError('Private source must be outside the checkout')
    public=verify(root);expected=json.loads((root/'PRIVATE_RUNTIME_REQUIREMENTS.json').read_text(encoding='utf-8'))
    original=source/'PACKAGE_MANIFEST.json'
    if sha(original)!=expected['source_package_manifest_sha256']:raise ValueError('Not the verified V17.3 source package')
    manifest=json.loads(original.read_text(encoding='utf-8'));files=manifest['files']
    if manifest.get('version')!='V17.3':raise ValueError('Wrong source version')
    if sha(source/'data/manifest.json')!=expected['data_manifest_sha256'] or sha(source/'extras/manifest.json')!=expected['extras_manifest_sha256']:raise ValueError('Wrong cohort manifests')
    for name,digest in files.items():
        parts=relative(name);p=source.joinpath(*parts.parts)
        if p.is_symlink() or not p.resolve().is_relative_to(source) or sha(p)!=digest:raise ValueError('Source file changed: '+name)
    picked=[]
    for name,digest in files.items():
        parts=relative(name)
        if parts.parts[0] not in ['data','extras','weights'] and name not in expected['source_snapshot_names']+['VALIDATION.json']:continue
        target=root.joinpath(*parts.parts)
        if not target.resolve().is_relative_to(root) or target.is_symlink():raise ValueError('Destination escapes checkout')
        if target.exists() and sha(target)!=digest:raise FileExistsError('Refusing different existing runtime file: '+name)
        picked.append((name,digest,target))
    # All source and existing destination files have been checked before copying.
    for name,digest,target in picked:
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source.joinpath(*relative(name).parts),target)
        if sha(target)!=digest:raise RuntimeError('Copy verification failed')
    runtime={name:digest for name,digest,target in picked};runtime.update(public)
    (root/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(version='V17.3-public-runtime',files=runtime),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'PRIVATE_RUNTIME_IMPORT.json').write_text(json.dumps(dict(state='complete',private_files=len(picked),source_package_manifest_sha256=sha(original),original_files_modified=False),indent=2)+'\n',encoding='utf-8')
    print('Private runtime verified and imported:',len(picked),'files. Source bundle unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args())
