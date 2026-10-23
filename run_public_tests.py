"""Run synthetic engineering tests without reading any cohort manifest or weights."""
from pathlib import Path
import sys,unittest,copy

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import torch
from threadpoolctl import threadpool_limits
import test_v172,test_v173,test_pipeline
from targets import SCHEMA

original_read=test_v172.read
def fixture_schema(path):
    # The original synthetic fixture takes only its schema from a private manifest.
    # Replace that one metadata lookup, never create a fake cohort file.
    if Path(path).resolve()==(ROOT/'data/manifest.json').resolve():return {'schema':copy.deepcopy(SCHEMA)}
    return original_read(path)

def main():
    test_v172.read=fixture_schema
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_pipeline,test_v172,test_v173])
    with threadpool_limits(limits=2):result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    torch.set_num_threads(2);main()
