param(
    [int]$Camera = 0
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$Root\realtime_infer.py" --source "$Camera" --config "$Root\config.yaml"
