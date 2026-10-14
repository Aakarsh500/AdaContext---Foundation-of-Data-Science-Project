$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonExecutable = if (Test-Path -LiteralPath $VirtualPython) { $VirtualPython } else { "python" }

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

Write-Host "Training the sample budget controller..."
& $PythonExecutable -m adacontext.cli train-budget --dataset sample --embedding-backend tfidf

Write-Host "Generating exploratory analysis..."
& $PythonExecutable -m adacontext.cli eda --dataset sample --embedding-backend tfidf

Write-Host "Evaluating all context-selection strategies..."
& $PythonExecutable -m adacontext.cli evaluate --dataset sample --split test --embedding-backend tfidf

Write-Host "Running the lost-in-the-middle stress test..."
& $PythonExecutable -m adacontext.cli stress-test --dataset sample --embedding-backend tfidf

Write-Host "Running automated tests..."
& $PythonExecutable -m pytest

Write-Host ""
Write-Host "All experiments completed. Results are under reports\generated."

