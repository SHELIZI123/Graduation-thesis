from pathlib import Path
import json, hashlib, csv
import numpy as np

MODALITIES = ('eeg', 'eye', 'heart')
FAMILIES = ('PSQI', 'SCL90', 'EPQ')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()

def new_dir(path, protected=()):
    path = Path(path).resolve()
    for value in protected:
        p = Path(value).resolve()
        if path == p or p in path.parents or path in p.parents:
            raise ValueError('Output overlaps protected input: ' + str(p))
    path.mkdir(parents=True, exist_ok=False)
    return path

def csvwrite(path, rows):
    if not rows: raise ValueError('Empty CSV')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    keep = np.isfinite(y) & np.isfinite(p); y, p = y[keep], p[keep]
    if not len(y): return {'n': 0, 'mae': None, 'rmse': None, 'r2': None, 'pearson': None}
    denom = np.sum((y - y.mean())**2)
    corr = float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and y.std() > 1e-12 and p.std() > 1e-12 else None
    return {'n': len(y), 'mae': float(np.mean(abs(y-p))), 'rmse': float(np.sqrt(np.mean((y-p)**2))),
            'r2': float(1-np.sum((y-p)**2)/denom) if denom > 1e-12 else None, 'pearson': corr}

def load_signals(root, ids=None):
    root = Path(root); meta = read(root/'manifest.json')
    if meta['state'] != 'complete': raise ValueError('Incomplete data')
    arrays = []
    selected=set(ids) if ids is not None else None
    if selected is not None and not selected<={e['subject_id'] for e in meta['subjects']}:raise KeyError('Unknown subject ID')
    for entry in meta['subjects']:
        if selected is not None and entry['subject_id'] not in selected:continue
        p = root/entry['file']
        if sha(p) != entry['sha256']: raise ValueError('Changed feature file')
        with np.load(p, allow_pickle=False) as z: arrays.append({m: z[m].copy() for m in MODALITIES})
    return meta, arrays

def load_data(root):
    root=Path(root);meta,arrays=load_signals(root)
    if sha(root/'targets.npz') != meta['targets_sha256']: raise ValueError('Changed targets')
    with np.load(root/'targets.npz', allow_pickle=False) as z: y = z['values'].copy()
    return meta, arrays, y
