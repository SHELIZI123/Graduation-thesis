"""Validate results path against all inputs BEFORE shell jobs create logs."""
import argparse
from pathlib import Path
from common import load_json,new_output
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--features',required=True);p.add_argument('--output',required=True)
args=p.parse_args()
root=Path(args.features).resolve(strict=True);cfg=load_json(root/'config.json')
new_output(args.output,root,cfg['raw_root'],cfg['processed_root'])
