"""Build a whitelist-based portable package; exclude local experiments."""
from pathlib import Path
import zipfile,hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    files=[]
    for directory in ['code','data','extras','weights','docs']:
        files += [p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    files += [ROOT/n for n in ['README.md','requirements.lock.txt','requirements.foundation.txt','requirements.txt',
        'run_server_dual_gpu.sh','run_smoke.sh','V171_SOURCE_SNAPSHOT.json','V172_SOURCE_SNAPSHOT.json','VALIDATION.json']]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sorted(files)}
    (ROOT/'PACKAGE_MANIFEST.json').write_text(json.dumps(dict(version='V17.3',files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
    files.append(ROOT/'PACKAGE_MANIFEST.json');target=ROOT.parent/'trimodal_v17_3_server_20261022.zip'
    with target.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in files:archive.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)).replace('\\','/'))
    checksum=digest(target);target.with_suffix('.zip.sha256').write_text(checksum+'  '+target.name+'\n',encoding='ascii')
    print(json.dumps(dict(file=str(target),bytes=target.stat().st_size,sha256=checksum,entries=len(files)),ensure_ascii=False))

if __name__=='__main__':main()
