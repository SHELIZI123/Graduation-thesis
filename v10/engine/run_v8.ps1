param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Features,
    [Parameter(Mandatory=$true)][string]$EmotionPlan,
    [Parameter(Mandatory=$true)][string]$PsqiPlan,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference='Stop'
& $Python -B "$PSScriptRoot\verify_package.py"
if ($LASTEXITCODE -ne 0) { throw 'Package checksum verification failed' }
& $Python -B "$PSScriptRoot\build_improved.py" --features $Features --plans $EmotionPlan $PsqiPlan --output $Output --devices cuda:0 --batch-size 16 --threads 2 --allow-unreviewed
if ($LASTEXITCODE -ne 0) { throw 'V8 failed; inspect the new output logs' }
