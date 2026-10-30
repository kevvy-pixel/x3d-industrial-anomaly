param(
    [Parameter(Mandatory = $true)][string]$VideoPath
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$Root\realtime_infer.py" --source $VideoPath --config "$Root\config.yaml"
