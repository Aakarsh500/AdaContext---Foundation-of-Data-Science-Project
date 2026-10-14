$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path -LiteralPath $VirtualEnvironment)) {
    python -m venv $VirtualEnvironment
}

$PythonExecutable = Join-Path $VirtualEnvironment "Scripts\python.exe"
& $PythonExecutable -m pip install --upgrade pip
& $PythonExecutable -m pip install -e "${ProjectRoot}[dev]"

Write-Host ""
Write-Host "Setup complete. Start the app with:"
Write-Host "  .\run_app.ps1"
