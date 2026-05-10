$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$stopScript = Join-Path $projectRoot "stop_ui.ps1"
$targetScript = "module1_ui.py"
$ports = @(8501, 8502, 8503, 8504, 8505)

if (Test-Path $stopScript) {
    powershell -ExecutionPolicy Bypass -File $stopScript
    Start-Sleep -Seconds 2
}

$selectedPort = $null

foreach ($port in $ports) {
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue

    if (-not $listener) {
        $selectedPort = $port
        break
    }
}

if (-not $selectedPort) {
    throw "No free port found in $($ports -join ', ')."
}

Start-Process `
    -FilePath python `
    -ArgumentList '-m', 'streamlit', 'run', $targetScript, '--server.port', "$selectedPort", '--server.headless', 'true' `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden

Start-Sleep -Seconds 6

$url = "http://127.0.0.1:$selectedPort"

try {
    $response = Invoke-WebRequest -UseBasicParsing $url
    Write-Output "Started $targetScript"
    Write-Output "URL: $url"
    Write-Output "HTTP status: $($response.StatusCode)"
}
catch {
    Write-Output "Started $targetScript, but the page is not reachable yet."
    Write-Output "URL: $url"
    Write-Output $_.Exception.Message
}
