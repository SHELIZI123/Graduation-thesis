"""Build a code/public-weight package, never include local participant outputs."""
from pathlib import Path
import json
import tarfile
import zipfile
from common import sha256


def main():
    root=Path(__file__).resolve().parent
    files=sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.ps1','.txt','.md'})
    for folder in ['vendor_cbramod','weights']:
        files += sorted(p for p in (root/folder).iterdir() if p.is_file() and p.suffix!='.part')
    if any(p.suffix in {'.npz','.npy','.csv','.edf','.joblib'} for p in files):
        raise ValueError('Participant artifact is forbidden in package')
    checks={p.relative_to(root).as_posix():sha256(p) for p in files}
    checkfile=root/'CHECKSUMS.json'
    checkfile.write_text(json.dumps(checks,indent=2),encoding='utf-8')
    files.append(checkfile)
    zip_path=root.with_suffix('.zip');tar_path=root.with_suffix('.tar.gz')
    if zip_path.exists() or tar_path.exists():raise FileExistsError('Refuse to overwrite release archives')
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files:archive.write(p,(Path(root.name)/p.relative_to(root)).as_posix())
    with tarfile.open(tar_path,'x:gz') as archive:
        for p in files:archive.add(p,arcname=(Path(root.name)/p.relative_to(root)).as_posix(),recursive=False)
    print(json.dumps({'files':len(files),'zip_bytes':zip_path.stat().st_size,'zip_sha256':sha256(zip_path),
        'tar_bytes':tar_path.stat().st_size,'tar_sha256':sha256(tar_path)},indent=2))


if __name__=='__main__':main()
