"""Observed questionnaire scores only. No item reconstruction or normative conversion."""
import math

def target(name, family, legacy, factor=None, low=None, high=None, weight=1., label=None):
    return dict(name=name, family=family, legacy=legacy, factor=factor, low=low, high=high,
                weight=weight, label=label or name, role='primary' if weight == 1 else 'auxiliary')

SCHEMA = [target('psqi_total', 'PSQI', 'zf', '总分', 0, 21, label='PSQI 总分')]
for key in ['zgsmzl','smqfq','smcxx','xgxsmxl','smwl','sysmyw','btgnwl']:
    SCHEMA.append(target('psqi_'+key, 'PSQI', key, low=0, high=3, weight=.2, label='PSQI 导出分项 '+key))
for key, name in [('qth','躯体化'),('qpzt','强迫'),('rjmg','人际关系'),('yy','抑郁'),('jl','焦虑'),
                  ('dd','敌对'),('kb','恐怖'),('pz','偏执'),('jsb','精神病性')]:
    SCHEMA.append(target('scl_'+key, 'SCL90', key+'_pj', name, 1, 5, label='SCL-90 '+name+'均分'))
SCHEMA.append(target('scl_mean', 'SCL90', 'zjf_pj', '总均分', 1, 5, label='SCL-90 总均分'))
SCHEMA += [target('scl_total','SCL90','zf_ys','总分',90,450,.15),
           target('scl_positive_count','SCL90','yaxs_ys','阳性项目数',0,90,.15),
           target('scl_negative_count','SCL90','yixs_ys','阴性项目数',0,90,.15),
           target('scl_positive_mean','SCL90','yxpj_pj','阳性项目均分',1,5,.15),
           target('scl_other','SCL90','qt_pj',low=1,high=5,weight=.15)]
for key, code, name in [('nw','E','外倾性（E）'),('sj','N','神经质（N）'),('js','P','精神质（P）'),('ys','L','效度（L）')]:
    SCHEMA.append(target('epq_'+code, 'EPQ', key+'_ys', name, low=0,
                         weight=.2 if code == 'L' else 1., label='EPQ '+name+'原始分'))

def finite(v):
    if v is None or isinstance(v, bool): return None
    try: v = float(v)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) else None

def parse(source):
    values, evidence, issues = [], [], []
    for t in SCHEMA:
        records = source['scales'][t['family']]
        value = None; reason = 'missing_scale' if not records else 'repeated_scale_visit_unmatched'
        if len(records) == 1:
            r = records[0]; a = finite(r['scores'].get(t['legacy']))
            ff = [f for f in r['factors'] if f['name'] == t['factor']] if t['factor'] else []
            b = finite(ff[0]['origin_score']) if len(ff) == 1 else None
            if len(ff) > 1: reason = 'duplicate_factor_name'
            elif a is not None and b is not None and abs(a-b) > 1e-4: reason = 'conflicting_export_fields'
            else:
                value = a if a is not None else b; reason = 'observed_legacy' if a is not None else 'observed_factor'
                if value is None: reason = 'field_not_exported'
                elif ((t['low'] is not None and value < t['low']) or
                      (t['high'] is not None and value > t['high'])):
                    value = None; reason = 'outside_declared_score_range'
        if t['name'].startswith('psqi_') and len(records) == 1:
            # Honor the original parser's component sum and total validation.
            if records[0]['observed_targets'].get('psqi') is None:
                value = None; reason = 'psqi_record_failed_total_validation'
        values.append(value)
        evidence.append({'target':t['name'], 'family':t['family'], 'value':value, 'source_field':t['legacy'],
                         'source_factor':t['factor'], 'status':reason, 'standard_score_used':False,
                         'imputed':False, 'visit_confirmed':False})
        if reason in ['conflicting_export_fields','duplicate_factor_name','outside_declared_score_range']:
            issues.append({'target':t['name'],'reason':reason})
    return values, evidence, issues
