from pathlib import Path
import argparse
import numpy as np
from common import load_data,write
from features_v171 import VIEWS,split_views,dimensions

def main(a):
    m,records,y=load_data(a.data);dd=dimensions(m['dimensions']);views={}
    for v in VIEWS:
        bags=[split_views(r)[v] for r in records];lengths=[len(b) for b in bags]
        nonempty=[b for b in bags if len(b)]
        allrows=np.concatenate(nonempty) if nonempty else np.empty((0,dd[v]))
        views[v]=dict(features=dd[v],available_subjects=sum(n>0 for n in lengths),
            record_count=int(sum(lengths)),bag_length_quantiles=np.quantile(lengths,[0,.5,.9,1]).tolist(),
            fraction_missing=float(1-np.isfinite(allrows).mean()) if len(allrows) else 1.,
            subjects_truncated_at_32=sum(n>32 for n in lengths),subjects_truncated_at_128=sum(n>128 for n in lengths))
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    write(out,dict(subjects=len(records),supervised=int(np.isfinite(y).any(1).sum()),views=views,
       scales_are_labels_only=True,labels_imputed=False,
       v17_psqi_total_loss_share=1/(1+7*.2),v171_psqi_total_loss_share=1/(1+7*.05),
       audit_scope='All-input descriptive audit. No cohort statistic from this file is consumed by training.',
       timing='Views share subject IDs only. PPG and NNI records have not been declared time-aligned.'))
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);main(p.parse_args())
