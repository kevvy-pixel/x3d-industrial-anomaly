$ErrorActionPreference = 'SilentlyContinue'
$Host.UI.RawUI.WindowTitle = 'X3D-M Binary Touch-Face Training Monitor'
$History = 'D:\CondaData\factory_x3d_m_touchface_binary\experiment\history.csv'
$BestMetrics = 'D:\CondaData\factory_x3d_m_touchface_binary\experiment\best_metrics.json'
$TargetEpochs = 30

while ($true) {
    Clear-Host
    Write-Host 'X3D-M Binary: normal vs touch_face' -ForegroundColor Cyan
    Write-Host ('Updated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host ''
    if (Test-Path -LiteralPath $History) {
        $Rows = @(Import-Csv -LiteralPath $History)
        Write-Host ("Completed: {0}/{1} epochs" -f $Rows.Count, $TargetEpochs) -ForegroundColor Yellow
        $Rows | Select-Object -Last 10 `
            @{N='epoch';E={$_.epoch}}, `
            @{N='train_loss';E={'{0:N4}' -f [double]$_.train_loss}}, `
            @{N='val_loss';E={'{0:N4}' -f [double]$_.val_loss}}, `
            @{N='accuracy';E={'{0:P2}' -f [double]$_.accuracy}}, `
            @{N='macro_f1';E={'{0:N4}' -f [double]$_.macro_f1}}, `
            @{N='pr_auc';E={'{0:N4}' -f [double]$_.pr_auc}}, `
            @{N='minutes';E={'{0:N1}' -f ([double]$_.seconds / 60)}} | Format-Table -AutoSize
        if (Test-Path -LiteralPath $BestMetrics) {
            $Best = Get-Content -LiteralPath $BestMetrics -Raw | ConvertFrom-Json
            Write-Host ("Best: epoch {0}, macro-F1 {1:N4}, accuracy {2:P2}" -f $Best.epoch, [double]$Best.macro_f1, [double]$Best.accuracy) -ForegroundColor Green
        }
        if ($Rows.Count -ge $TargetEpochs) {
            Write-Host ''
            Write-Host 'Training complete. This window may be closed.' -ForegroundColor Green
            break
        }
    } else {
        Write-Host 'Waiting for history.csv ...' -ForegroundColor Yellow
    }
    Write-Host ''
    Write-Host 'GPU:' -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
    Write-Host ''
    Write-Host 'Refresh: 5 seconds. Closing this monitor does not stop training.'
    Start-Sleep -Seconds 5
}
