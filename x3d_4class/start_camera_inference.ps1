$ErrorActionPreference = 'Stop'
$python = 'python'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root 'infer_stream.py'
$model = Join-Path $root 'experiment\x3d_m_4class.ts'

& $python $script `
    --model $model `
    --source 0 `
    --stride-sec 0.3 `
    --smooth 3 `
    --board-threshold 0.60 `
    --single-threshold 0.60 `
    --touch-threshold 0.65 `
    --merge-gap-sec 2.0 `
    --display

Write-Host ''
Write-Host 'Inference stopped. Press Enter to close.'
Read-Host
