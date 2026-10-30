$ErrorActionPreference = 'SilentlyContinue'
$host.UI.RawUI.WindowTitle = 'X3D-M Training Monitor'
$history = 'D:\CondaData\factory_x3d_m\experiment\history.csv'
$targetEpochs = 30

while ($true) {
    Clear-Host
    Write-Host 'X3D-M Four-Class Training Monitor' -ForegroundColor Cyan
    Write-Host ('Updated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host ''
    if (Test-Path -LiteralPath $history) {
        $rows = @(Import-Csv -LiteralPath $history)
        Write-Host ("Completed: {0}/{1} epochs" -f $rows.Count, $targetEpochs) -ForegroundColor Yellow
        $rows | Select-Object -Last 8 `
            @{N='epoch';E={$_.epoch}}, `
            @{N='train_loss';E={'{0:N4}' -f [double]$_.train_loss}}, `
            @{N='val_loss';E={'{0:N4}' -f [double]$_.val_loss}}, `
            @{N='accuracy';E={'{0:P2}' -f [double]$_.accuracy}}, `
            @{N='macro_f1';E={'{0:N4}' -f [double]$_.macro_f1}}, `
            @{N='minutes';E={'{0:N1}' -f ([double]$_.seconds / 60)}} | Format-Table -AutoSize
        if ($rows.Count -ge $targetEpochs) {
            Write-Host 'Training complete.' -ForegroundColor Green
            Write-Host 'Best model: D:\CondaData\factory_x3d_m\experiment\best.pt'
            Write-Host 'Press any key to close.'
            $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
            break
        }
    } else {
        Write-Host 'Waiting for history.csv ...' -ForegroundColor Yellow
    }
    Write-Host 'GPU:' -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
    Write-Host ''
    Write-Host 'Refreshes every 5 seconds. Closing this window does not stop training.'
    Start-Sleep -Seconds 5
}
