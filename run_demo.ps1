$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonExecutable = if (Test-Path -LiteralPath $VirtualPython) { $VirtualPython } else { "python" }

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $PythonExecutable -m adacontext.cli demo --dataset sample --embedding-backend tfidf

