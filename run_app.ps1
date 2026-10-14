$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonExecutable = if (Test-Path -LiteralPath $VirtualPython) { $VirtualPython } else { "python" }

Set-Location -LiteralPath $ProjectRoot
& $PythonExecutable -m streamlit run app.py

