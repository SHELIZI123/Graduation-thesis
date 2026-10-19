"""Read raw files; write a NEW processed directory with source integrity audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import new_output, save_json, save_csv, csv_rows, yes
from signal_io import DEFAULT, inventory, process_eeg, process_eye, process_scales


def main(args):
    source = Path(args.source).resolve(strict=True)
    out = new_output(args.output, source)
    before = inventory(source)
    save_json(out/'audit/source_before.json', before)
    ids = sorted(p for p in source.iterdir() if p.is_dir() and p.name.isdigit())
    if args.ids:
        wanted = set(args.ids)
        if wanted - {p.name for p in ids}:
            raise ValueError('Unknown IDs')
        ids = [p for p in ids if p.name in wanted]
    selections = {r['subject_id']: r['eye_directory'] for r in csv_rows(args.visit_selection)} if args.visit_selection else {}
    subjects, tasks, review, errors = [], [], [], []
    try:
        for i, directory in enumerate(ids):
            sid = directory.name
            dest = out/'subjects'/sid
            dest.mkdir(parents=True)
            row = {'subject_id': sid, 'eeg_windows': 0, 'eye_windows': 0, 'hrv_records': 0}
            cfg = dict(DEFAULT)
            if sid in selections:
                chosen = (directory/selections[sid]).resolve(strict=True)
                if directory.resolve() not in chosen.parents:
                    raise ValueError('Unsafe eye directory selection')
                cfg['eye_directory'] = selections[sid]
            for modality in ['eeg', 'eye', 'scales']:
                try:
                    if modality == 'eeg':
                        result, qc = process_eeg(directory, dest, cfg)
                        save_json(dest/'eeg_qc.json', qc)
                    elif modality == 'eye':
                        if len(list(directory.rglob('vpr_samples.csv'))) > 1 and sid not in selections:
                            raise ValueError('Multiple eye recordings: fill --visit-selection; no automatic visit choice')
                        result, qc, mapping = process_eye(directory, dest, cfg)
                        save_json(dest/'eye_qc.json', qc)
                        for t in mapping:
                            t.update(task_type='unknown', coordinate_frame_confirmed='', task_confirmed='')
                        tasks.extend(mapping)
                    else:
                        result, _ = process_scales(directory, dest)
                    row.update(result)
                except (ValueError, KeyError, OSError) as e:
                    errors.append({'subject_id': sid, 'modality': modality, 'error': str(e)})
                    row[modality+'_status'] = 'excluded_needs_review'
            # NNI is stored without interpreting its units or selecting among visits.
            hrv = []
            for path in directory.rglob('hrv_nni_sequences.jsonl'):
                for line in path.read_text(encoding='utf-8-sig').splitlines():
                    value = json.loads(line)
                    if value['record_id'] != sid:
                        raise ValueError('HRV subject mismatch')
                    hrv.append({'record_sequence': value['record_sequence'], 'nni': value['nni']})
            save_json(dest/'nni_records.json', hrv)
            row['hrv_records'] = len(hrv)
            row.setdefault('targets', {'psqi': None, 'depression': None, 'anxiety': None})
            row.setdefault('age', None)
            row.setdefault('sex', None)
            subjects.append(row)
            review.append({'subject_id': sid, 'group_id': sid, 'include': '', 'identity_confirmed': '',
                           'visit_confirmed': '', 'eeg_protocol_confirmed': '', 'eye_protocol_confirmed': '',
                           'hrv_record_sequence': '', 'nni_unit': '', 'notes': ''})
            if (i+1) % 20 == 0 or i+1 == len(ids):
                print(f'Processed {i+1}/{len(ids)}; exclusions {len(errors)}', flush=True)
        save_json(out/'manifest.json', {'format_version': 3, 'subjects': subjects})
        save_json(out/'processing_config.json', {'source_root': str(source), 'parameters': DEFAULT,
                    'source_read_only': True, 'eye_selection': selections,
                    'all_quality_thresholds_are_engineering_heuristics': True})
        save_csv(out/'review/subjects.csv', review)
        save_csv(out/'review/eye_tasks.csv', tasks, list(tasks[0]) if tasks else
                 ['subject_id', 'measurement_id', 'task_type', 'coordinate_frame_confirmed', 'task_confirmed'])
        save_json(out/'audit/exclusions.json', errors)
        save_json(out/'summary.json', {'ids': len(subjects), 'eeg_usable': sum(s['eeg_windows']>0 for s in subjects),
                    'eye_usable': sum(s['eye_windows']>0 for s in subjects),
                    'hrv_present': sum(s['hrv_records']>0 for s in subjects), 'exclusions': len(errors),
                    'status': 'review_required', 'no_clinical_diagnosis_labels': True})
    finally:
        after = inventory(source)
        save_json(out/'audit/source_integrity.json', {'unchanged': before == after, 'file_count': len(after),
                    'checks': ['paths', 'size', 'mtime_ns', 'sha256']})
        if before != after:
            raise RuntimeError('Source changed during processing; inspect audit')
    print('Prepared:', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ids', nargs='+')
    p.add_argument('--visit-selection', help='CSV subject_id,eye_directory; path relative to each ID')
    main(p.parse_args())
