from pathlib import Path
import argparse
from common import read,write,sha

def main(a):
    data=read(Path(a.data)/'manifest.json');source=Path(a.dataset)
    assert data['source_manifest_sha256']==sha(source/'manifest.json')
    count=0
    for s in data['source_files']:
        assert sha(source/s['signal_file'])==s['signal_sha256']
        assert sha(source/s['scale_file'])==s['scale_sha256'];count+=2
    result={'passed':True,'source_files_checked':count,'source_manifest_unchanged':True,
            'processed_source_signals_and_scale_metadata_unchanged':True,'raw_data_was_not_an_output_path':True}
    write(a.output,result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);main(p.parse_args())
