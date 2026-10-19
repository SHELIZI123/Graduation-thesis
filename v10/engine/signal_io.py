"""Signal/scale helpers adapted from the locally validated 2026-10-08 preprocessor.
No CLI or import-time writes; callers supply a new destination directory.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy import signal

CHANNELS = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6']
DEFAULT = {
    'eeg_output_hz': 200, 'eeg_bandpass_hz': [1., 45.], 'eeg_filter_order': 4,
    'eeg_reference': 'unchanged_from_source', 'eeg_window_seconds': 4,
    'eeg_patch_samples': 200, 'eeg_edge_exclusion_seconds': 2,
    'eeg_input_divisor_uv': 100., 'eeg_min_channel_std_uv': 0.5,
    'eeg_max_channel_std_uv': 150., 'eeg_max_abs_uv': 500.,
    'eeg_max_peak_to_peak_uv': 1000., 'reject_eeg_event_transition_windows': True,
    'eye_window_seconds': 4., 'eye_model_samples': 250,
    'eye_model_grid_hz': 62.5, 'eye_min_valid_fraction': 0.8,
    'eye_max_interpolation_gap_seconds': 0.1,
    'eye_channels': ['channel1_x', 'channel1_y'], 'eye_coordinate_unit': 'unconfirmed',
    'minimum_eeg_windows_for_pilot': 3, 'minimum_eye_windows_for_pilot': 1,
    'split_seed': 20261008, 'pilot_target': 'psqi',
    'modalities_synchronized': False, 'pupil_measurements_available': False,
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def number_or_nan(value):
    result = finite_number(value)
    return np.nan if result is None else result


def inventory(root):
    result = []
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                digest.update(chunk)
        result.append({'path': path.relative_to(root).as_posix(), 'bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns, 'sha256': digest.hexdigest()})
    return result


def read_edf(path):
    with path.open('rb') as stream:
        header = stream.read(256)
        n = int(header[252:256])
        header_bytes = int(header[184:192])
        records = int(header[236:244])
        record_seconds = float(header[244:252])
        packed = stream.read(n * 256)
        fields, offset = [], 0
        for width in [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]:
            fields.append([packed[offset+i*width:offset+(i+1)*width].decode('ascii', 'strict').strip()
                           for i in range(n)])
            offset += n * width
        stream.seek(header_bytes)
        payload = stream.read()
    samples = np.array([int(x) for x in fields[8]])
    if records <= 0 or record_seconds <= 0 or len(payload) != records * samples.sum() * 2:
        raise ValueError('Unsupported or inconsistent EDF payload length.')
    raw = np.frombuffer(payload, dtype='<i2').reshape(records, int(samples.sum()))
    starts = np.r_[0, np.cumsum(samples)]
    digital = [raw[:, starts[i]:starts[i+1]].reshape(-1) for i in range(n)]
    physical_min, physical_max = np.array(fields[3], float), np.array(fields[4], float)
    digital_min, digital_max = np.array(fields[5], float), np.array(fields[6], float)
    if np.any(digital_max <= digital_min):
        raise ValueError('Invalid EDF digital calibration limits.')
    return {
        'labels': fields[0], 'units': fields[2], 'digital': digital,
        'rates': samples / record_seconds, 'duration_seconds': records * record_seconds,
        'physical_min': physical_min, 'physical_max': physical_max,
        'digital_min': digital_min, 'digital_max': digital_max,
        'payload_sha256': hashlib.sha256(payload).hexdigest(),
    }


def process_eeg(source, destination, cfg):
    files = sorted((source / '脑电').glob('*.edf'))
    if not files:
        return {'eeg_status': 'missing', 'eeg_windows': 0}, []
    if len(files) != 1:
        raise ValueError('Multiple EEG files require visit matching; none selected automatically.')
    data = read_edf(files[0])
    records = read_csv(source / '脑电' / 'eeg_records.csv')
    if len(records) == 1 and records[0].get('waveform_sha256'):
        if data['payload_sha256'] != records[0]['waveform_sha256']:
            raise ValueError('EDF payload hash disagrees with source export manifest.')
    labels = [label.upper() for label in data['labels']]
    if any(labels.count(name) != 1 for name in CHANNELS):
        raise ValueError('Missing or ambiguous required EEG channel.')
    indices = [labels.index(name) for name in CHANNELS]
    rates = data['rates'][indices]
    if not np.allclose(rates, rates[0]):
        raise ValueError('EEG channels have unequal sampling rates.')
    native_hz = float(rates[0])
    units = {'uv': 1., 'µv': 1., 'mv': 1000., 'v': 1e6}
    native = []
    for i in indices:
        unit = data['units'][i].lower()
        if unit not in units:
            raise ValueError('EEG physical unit is unsupported: ' + unit)
        scale = (data['physical_max'][i] - data['physical_min'][i]) / (data['digital_max'][i] - data['digital_min'][i])
        native.append(((data['digital'][i].astype(np.float64) - data['digital_min'][i]) * scale
                       + data['physical_min'][i]) * units[unit])
    native = np.stack(native)
    if not np.isfinite(native).all():
        raise ValueError('Nonfinite EEG samples.')
    fraction = Fraction(cfg['eeg_output_hz'] / native_hz).limit_denominator(10000)
    downsampled = signal.resample_poly(native, fraction.numerator, fraction.denominator, axis=-1)
    sos = signal.butter(cfg['eeg_filter_order'], cfg['eeg_bandpass_hz'], btype='bandpass',
                        fs=cfg['eeg_output_hz'], output='sos')
    filtered = signal.sosfiltfilt(sos, downsampled, axis=-1).astype(np.float32)
    if not np.isfinite(filtered).all():
        raise ValueError('Nonfinite filtered EEG samples.')
    np.save(destination / 'eeg_200hz_uv.npy', filtered, allow_pickle=False)
    events = []
    for i, name in enumerate(data['labels']):
        if i in indices:
            continue
        a = data['digital'][i]
        changes = np.flatnonzero(a[1:] != a[:-1]) + 1
        change_times = changes / data['rates'][i]
        run_starts = np.r_[0, changes]
        run_ends = np.r_[changes, len(a)]
        events.append({'label': name, 'semantics': 'unconfirmed',
                       'transition_seconds': change_times.tolist(),
                       'runs': [{'start_s': float(s / data['rates'][i]),
                                 'end_s': float(e / data['rates'][i]), 'digital_value': int(a[s])}
                                for s, e in zip(run_starts, run_ends)]})
    write_json(destination / 'eeg_events.json', events)
    windows, metadata, qc = [], [], []
    length = cfg['eeg_window_seconds']
    duration = min(data['duration_seconds'], filtered.shape[-1] / cfg['eeg_output_hz'])
    start = float(cfg['eeg_edge_exclusion_seconds'])
    while start + length <= duration - cfg['eeg_edge_exclusion_seconds'] + 1e-9:
        end = start + length
        a, b = int(round(start * native_hz)), int(round(end * native_hz))
        segment = native[:, a:b]
        first = int(round(start * cfg['eeg_output_hz']))
        filtered_segment = filtered[:, first:first + length * cfg['eeg_output_hz']]
        std = segment.std(axis=1)
        peak = float(np.abs(segment).max())
        ptp = float(np.ptp(segment, axis=1).max())
        saturated = any(np.any((data['digital'][i][a:b] <= data['digital_min'][i])
                              | (data['digital'][i][a:b] >= data['digital_max'][i])) for i in indices)
        transitions = any(any(start < t < end for t in event['transition_seconds']) for event in events)
        reasons = []
        if float(std.min()) < cfg['eeg_min_channel_std_uv']:
            reasons.append('low_variance')
        if float(std.max()) > cfg['eeg_max_channel_std_uv']:
            reasons.append('high_variance')
        if peak > cfg['eeg_max_abs_uv']:
            reasons.append('large_amplitude')
        if ptp > cfg['eeg_max_peak_to_peak_uv']:
            reasons.append('large_peak_to_peak')
        if saturated:
            reasons.append('digital_saturation')
        if float(np.abs(filtered_segment).max()) > cfg['eeg_max_abs_uv']:
            reasons.append('filtered_large_amplitude')
        if float(filtered_segment.std(axis=1).max()) > cfg['eeg_max_channel_std_uv']:
            reasons.append('filtered_high_variance')
        if transitions and cfg['reject_eeg_event_transition_windows']:
            reasons.append('event_transition')
        states = {event['label']: next(run['digital_value'] for run in event['runs']
                  if run['start_s'] <= start < run['end_s']) for event in events}
        row = {'subject_id': source.name, 'start_s': start, 'end_s': end,
               'accepted': not reasons, 'reasons': '|'.join(reasons), 'min_std_uv': float(std.min()),
               'max_std_uv': float(std.max()), 'max_abs_uv': peak, 'max_ptp_uv': ptp,
               'event_states_json': json.dumps(states, ensure_ascii=False)}
        qc.append(row)
        if not reasons:
            block = filtered_segment
            block = (block / cfg['eeg_input_divisor_uv']).reshape(16, length, cfg['eeg_patch_samples'])
            windows.append(block)
            metadata.append({'window_index': len(windows) - 1, **row})
        start += length
    array = np.stack(windows).astype(np.float32) if windows else np.empty((0, 16, length, 200), np.float32)
    np.save(destination / 'eeg_windows.npy', array, allow_pickle=False)
    write_csv(destination / 'eeg_window_metadata.csv', metadata,
              ['window_index', 'subject_id', 'start_s', 'end_s', 'accepted', 'reasons',
               'min_std_uv', 'max_std_uv', 'max_abs_uv', 'max_ptp_uv', 'event_states_json'])
    write_json(destination / 'eeg_metadata.json', {
        'source_file': files[0].relative_to(source).as_posix(), 'channels': CHANNELS,
        'native_hz': native_hz, 'processed_hz': cfg['eeg_output_hz'], 'duration_seconds': duration,
        'continuous_unit': 'uV', 'model_input_unit': 'uV / 100', 'events_are_model_inputs': False,
        'reference': cfg['eeg_reference'], 'shape': list(array.shape),
    })
    return {'eeg_status': 'processed', 'eeg_windows': len(array), 'eeg_candidate_windows': len(qc),
            'eeg_native_hz': native_hz, 'eeg_duration_s': duration,
            'eeg_path': f'subjects/{source.name}/eeg_windows.npy'}, qc


def process_eye(source, destination, cfg):
    samples = read_csv(source / cfg.get('eye_directory', '眼动') / 'vpr_samples.csv')
    measures = read_csv(source / cfg.get('eye_directory', '眼动') / 'vpr_measurements.csv')
    if not samples:
        return {'eye_status': 'missing', 'eye_windows': 0}, [], []
    grouped = collections.defaultdict(list)
    for row in samples:
        if row['record_id'] != source.name:
            raise ValueError('Eye row subject ID mismatch.')
        grouped[row['measurement_id']].append(row)
    measure_lookup = {row['measurement_id']: row for row in measures}
    native_dir = destination / 'eye_native'
    native_dir.mkdir(exist_ok=True)
    accepted_windows, masks, metadata, qc, tasks = [], [], [], [], []
    for mid, rows in sorted(grouped.items()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', mid):
            raise ValueError('Unsafe measurement filename.')
        t = np.array([number_or_nan(row['time_s']) for row in rows], np.float64)
        c1 = np.array([[number_or_nan(row[f'channel1_{axis}']) for axis in 'xyz'] for row in rows], np.float32)
        c2 = np.array([[number_or_nan(row[f'channel2_{axis}']) for axis in 'xyz'] for row in rows], np.float32)
        stimulus = np.array([[number_or_nan(row[f'stimulus_{axis}']) for axis in 'xyz'] for row in rows], np.float32)
        flag1 = np.array([row.get('channel1_valid') == '1' for row in rows])
        flag2 = np.array([row.get('channel2_valid') == '1' for row in rows])
        valid = flag1 & np.isfinite(c1[:, :2]).all(axis=1) & np.isfinite(t)
        np.savez_compressed(native_dir / (mid + '.npz'), time_s=t, channel1_xyz=c1,
                            channel2_xyz=c2, channel1_valid=flag1, channel2_valid=flag2,
                            stimulus_xyz=stimulus)
        declared_fs = finite_number(measure_lookup.get(mid, {}).get('samplerate_hz'))
        time_ok = np.isfinite(t).all() and len(t) >= 2 and np.all(np.diff(t) > 0)
        fs = float(1. / np.median(np.diff(t))) if time_ok else None
        task = {'subject_id': source.name, 'measurement_id': mid, 'task_name': '',
                'task_mapping_status': 'unconfirmed', 'samples': len(t),
                'declared_hz': declared_fs, 'inferred_hz': fs,
                'channel1_valid_fraction': float(valid.mean()), 'accepted_windows': 0}
        tasks.append(task)
        if not time_ok or fs is None or fs <= 0:
            qc.append({'subject_id': source.name, 'measurement_id': mid, 'start_s': None,
                       'accepted': False, 'reasons': 'invalid_time_axis', 'valid_fraction': float(valid.mean())})
            continue
        origin = float(t[0])
        window_seconds = cfg['eye_window_seconds']
        count = int(math.floor((t[-1] - origin + 1. / fs + 1e-8) / window_seconds))
        if count == 0:
            qc.append({'subject_id': source.name, 'measurement_id': mid, 'start_s': origin,
                       'accepted': False, 'reasons': 'too_short', 'valid_fraction': float(valid.mean())})
        good_times = t[valid]
        good_xy = c1[valid, :2]
        for window in range(count):
            start = origin + window * window_seconds
            end = start + window_seconds
            inside = (t >= start) & (t < end)
            valid_fraction = float(valid[inside].mean()) if inside.any() else 0.
            grid = start + np.arange(cfg['eye_model_samples']) / cfg['eye_model_grid_hz']
            # Extra dense-grid points for downsampling are used only within the same measurement/window.
            processing_grid = (start + np.arange(int(round(window_seconds * fs))) / fs
                               if fs > cfg['eye_model_grid_hz'] * 1.01 else grid)
            reasons = []
            if valid_fraction < cfg['eye_min_valid_fraction']:
                reasons.append('low_valid_fraction')
            if len(good_times) < 2 or good_times[0] > processing_grid[0] + 1e-8 or good_times[-1] < processing_grid[-1] - 1e-8:
                reasons.append('missing_edge_support')
            if not reasons:
                right = np.clip(np.searchsorted(good_times, processing_grid), 0, len(good_times) - 1)
                left = np.clip(right - 1, 0, len(good_times) - 1)
                gaps = good_times[right] - good_times[left]
                gaps[np.isclose(good_times[right], processing_grid, rtol=0, atol=1e-8)] = 0
                if np.any(gaps > cfg['eye_max_interpolation_gap_seconds']):
                    reasons.append('long_invalid_or_time_gap')
            row = {'subject_id': source.name, 'measurement_id': mid, 'start_s': start,
                   'end_s': end, 'accepted': not reasons, 'reasons': '|'.join(reasons),
                   'valid_fraction': valid_fraction, 'native_hz': fs}
            qc.append(row)
            if reasons:
                continue
            xy = np.stack([np.interp(processing_grid, good_times, good_xy[:, i]) for i in range(2)])
            if fs > cfg['eye_model_grid_hz'] * 1.01:
                cutoff = 0.45 * cfg['eye_model_grid_hz']
                lowpass = signal.butter(4, cutoff, fs=fs, output='sos')
                xy = signal.sosfiltfilt(lowpass, xy, axis=-1)
                xy = np.stack([np.interp(grid, processing_grid, channel) for channel in xy])
            # Save original validity sampled at the nearest original timestamp, even after interpolation.
            right = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
            left = np.clip(right - 1, 0, len(t) - 1)
            nearest = np.where(np.abs(t[right] - grid) < np.abs(t[left] - grid), right, left)
            accepted_windows.append(xy.astype(np.float32).reshape(2, 1, 250))
            masks.append(valid[nearest].reshape(1, 250))
            task['accepted_windows'] += 1
            metadata.append({'window_index': len(accepted_windows) - 1, **row,
                             'task_name': '', 'task_mapping_status': 'unconfirmed'})
    array = np.stack(accepted_windows) if accepted_windows else np.empty((0, 2, 1, 250), np.float32)
    mask_array = np.stack(masks) if masks else np.empty((0, 1, 250), bool)
    np.save(destination / 'eye_windows.npy', array, allow_pickle=False)
    np.save(destination / 'eye_valid_masks.npy', mask_array, allow_pickle=False)
    write_csv(destination / 'eye_window_metadata.csv', metadata,
              ['window_index', 'subject_id', 'measurement_id', 'start_s', 'end_s', 'accepted',
               'reasons', 'valid_fraction', 'native_hz', 'task_name', 'task_mapping_status'])
    write_json(destination / 'eye_metadata.json', {
        'model_channels': cfg['eye_channels'], 'coordinate_unit': 'unconfirmed',
        'grid_hz': cfg['eye_model_grid_hz'], 'window_seconds': cfg['eye_window_seconds'],
        'eye_patch_duration_seconds': cfg['eye_window_seconds'],
        'pupil_synthesized': False, 'model_uses_channel2': False,
        'model_uses_stimulus_or_z': False, 'shape': list(array.shape),
        'normalization': 'none; fit any dataset-level scaler in the training fold only',
    })
    return {'eye_status': 'processed', 'eye_windows': len(array), 'eye_measurements': len(tasks),
            'eye_path': f'subjects/{source.name}/eye_windows.npy',
            'eye_mask_path': f'subjects/{source.name}/eye_valid_masks.npy'}, qc, tasks


def process_scales(source, destination):
    path = source / '量表' / '量表数据.json'
    source_records = json.loads(path.read_text(encoding='utf-8-sig')).get('questionnaires', {}) if path.exists() else {}
    sanitized, flat, ages, sexes = {}, [], set(), set()
    targets = {'psqi': None, 'depression': None, 'anxiety': None}
    statuses = {}
    psqi_keys = {'zf', 'zgsmzl', 'smqfq', 'smcxx', 'xgxsmxl', 'smwl', 'sysmyw', 'btgnwl'}
    epq_keys = {'js_ys', 'js_bz', 'nw_ys', 'nw_bz', 'sj_ys', 'sj_bz', 'ys_ys', 'ys_bz'}
    scl_prefixes = {'zf', 'zjf', 'yixs', 'yaxs', 'yxpj', 'qth', 'qpzt', 'rjmg', 'yy', 'jl', 'dd', 'kb', 'pz', 'jsb', 'qt'}
    date_values = []
    for name, records in source_records.items():
        scale = {'PSQI': 'PSQI', '症状自评量表': 'SCL90', '艾森克人格量表': 'EPQ'}.get(name)
        if scale is None:
            continue
        sanitized[scale] = []
        statuses[scale] = 'missing' if not records else 'single_record' if len(records) == 1 else 'repeated_needs_visit_matching'
        for index, record in enumerate(records):
            scores = {}
            for key, value in record.items():
                if ((scale == 'PSQI' and key in psqi_keys) or (scale == 'EPQ' and key in epq_keys)
                    or (scale == 'SCL90' and key.split('_')[0] in scl_prefixes and key.endswith(('_ys', '_pj')))):
                    scores[key] = finite_number(value)
            items = {}
            for key, value in record.items():
                if re.fullmatch(r'saw\d+', key, re.I):
                    numeric = finite_number(value)
                    if numeric is not None:
                        items[key.lower()] = numeric
                    elif scale == 'PSQI' and isinstance(value, str) and re.fullmatch(r'\d{1,2}:\d{2}:\d{2}', value):
                        items[key.lower()] = value
                    else:
                        items[key.lower()] = None
            age = finite_number(record.get('age'))
            if age is not None and 0 < age <= 120:
                ages.add(age)
            sex = record.get('sex')
            if sex in {'男', '女'}:
                sexes.add(sex)
            date = finite_number(record.get('date'))
            if date is not None:
                date_values.append(date)
            clean = {'record_index': index, 'date_excel': date, 'scores': scores, 'items': items,
                     'age': age, 'sex': sex if sex in {'男', '女'} else None}
            sanitized[scale].append(clean)
            flat.append({'subject_id': source.name, 'scale': scale, 'record_index': index,
                         'date_excel': date, 'psqi': scores.get('zf') if scale == 'PSQI' else None,
                         'depression': scores.get('yy_pj'), 'anxiety': scores.get('jl_pj')})
        if len(records) == 1:
            scores = sanitized[scale][0]['scores']
            if scale == 'PSQI':
                value = scores.get('zf')
                components = [scores.get(key) for key in ['zgsmzl', 'smqfq', 'smcxx', 'xgxsmxl', 'smwl', 'sysmyw', 'btgnwl']]
                if value is not None and 0 <= value <= 21 and all(v is not None and 0 <= v <= 3 for v in components) and abs(sum(components) - value) < 1e-6:
                    targets['psqi'] = value
                else:
                    statuses[scale] = 'invalid_score_or_component_sum'
            if scale == 'SCL90':
                for target, key in [('depression', 'yy_pj'), ('anxiety', 'jl_pj')]:
                    value = scores.get(key)
                    if value is not None and 1 <= value <= 5:
                        targets[target] = value
    write_json(destination / 'scales_sanitized.json', {'subject_id': source.name, 'scales': sanitized})
    return {'targets': targets, 'scale_statuses': statuses,
            'age': next(iter(ages)) if len(ages) == 1 else None, 'age_conflict': len(ages) > 1,
            'sex': next(iter(sexes)) if len(sexes) == 1 else None,
            'scale_date_span_days': max(date_values) - min(date_values) if date_values else None,
            'visit_match_status': 'unconfirmed', 'clinical_diagnosis': None}, flat

