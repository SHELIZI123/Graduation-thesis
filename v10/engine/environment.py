"""Print environment details for reproducibility; no downloads or installs."""
import json,platform,sys,importlib.metadata
report={'python':sys.version,'platform':platform.platform()}
for name in ['numpy','scipy','scikit-learn','torch','tabpfn','joblib','threadpoolctl']:
    try:report[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:report[name]=None
try:
    import torch
    report['cuda_available']=torch.cuda.is_available()
    report['cuda_runtime']=torch.version.cuda
    report['gpus']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except ImportError:report['cuda_available']=False
print(json.dumps(report,ensure_ascii=False,indent=2))
