param(
  [string]$Source = $env:WORKER_RTSP_URL,
  [string]$DeviceId = $env:WORKER_DEVICE_ID,
  [string]$CallbackUrl = $env:WORKER_CALLBACK_URL,
  [switch]$Display
)
if (-not $Source) { throw 'Set WORKER_RTSP_URL or pass -Source.' }
$args = @('-m','worker_behavior.delivery','--source',$Source)
if ($DeviceId) { $args += @('--device-id',$DeviceId) }
if ($CallbackUrl) { $args += @('--callback-url',$CallbackUrl) }
if ($Display) { $args += '--display' }
python @args
