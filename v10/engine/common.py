"""Small shared utilities; no filesystem mutation on import."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def new_output(path, *protected):
    p = Path(path).resolve()
    for source in protected:
        if source is None:
            continue
        s = Path(source).resolve()
        if p == s or p in s.parents or s in p.parents:
            raise ValueError('Output must be separate from protected input: ' + str(s))
    p.mkdir(parents=True, exist_ok=False)
    return p


def yes(value):
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def metrics(y, p):
    return {'mae': float(mean_absolute_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'r2': float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else None}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def manifest(root):
    return {r['subject_id']: r for r in load_json(Path(root)/'manifest.json')['subjects']}


def validate_folds(plan):
    groups = plan['groups']
    seen = []
    for fold in plan['folds']:
        train, test = set(fold['development']), set(fold['test'])
        assert not train & test
        assert not {groups[x] for x in train} & {groups[x] for x in test}
        assert train | test == set(plan['subject_ids'])
        validation_seen = []
        for inner in fold['inner']:
            a, b = set(inner['train']), set(inner['validation'])
            assert a | b == train and not a & b
            assert not {groups[x] for x in a} & {groups[x] for x in b}
            validation_seen.extend(inner['validation'])
        assert sorted(validation_seen) == sorted(train)
        seen.extend(fold['test'])
    assert sorted(seen) == sorted(plan['subject_ids'])
    assert not set(seen) & set(plan.get('reserved_ids', []))
