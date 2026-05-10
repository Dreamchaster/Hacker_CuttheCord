$ErrorActionPreference = "Stop"

$targetScript = "module1_ui.py"
$pythonNames = @("python.exe", "pythonw.exe")

$processes = Get-CimInstance Win32_Process |
    Where-Object {
        ($pythonNames -contains $_.Name) -and
        $_.CommandLine -like "*$targetScript*"
    }

if (-not $processes) {
    Write-Output "No running $targetScript process found."
    exit 0
}

foreach ($process in $processes) {
    try {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
        Write-Output "Stopped PID $($process.ProcessId)"
    }
    catch {
        Write-Output "Failed to stop PID $($process.ProcessId): $($_.Exception.Message)"
    }
}
