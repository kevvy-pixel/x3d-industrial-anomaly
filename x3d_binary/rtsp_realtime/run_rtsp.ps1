param(
    [Parameter(Mandatory = $true)]
    [string]$RtspUrl,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

if (-not (Get-Command -Name $PythonExe -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Pass -PythonExe with the full path to python.exe."
}

& $PythonExe "$ProjectDir\realtime_infer.py" `
    --source $RtspUrl `
    --config "$ProjectDir\config.yaml"
