# Starts the API (new window) and the Vite dev server (this window).
# Run from anywhere: .\scripts\dev.ps1   then open http://localhost:5173
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path .venv\Scripts\python.exe)) { throw "No .venv: run scripts\setup_venv.ps1 first." }
if (-not (Test-Path web\node_modules)) { throw "No web\node_modules: run 'npm --prefix web install' first." }
if (-not (Test-Path .env)) { Write-Warning "No .env found: using defaults (copy .env.example to .env to change them)." }

$python = Join-Path $root ".venv\Scripts\python.exe"
Start-Process -FilePath $python -ArgumentList "-m", "voicebot.web" -WorkingDirectory (Join-Path $root "server")

Write-Host "API starting in a new window (first start loads the models; watch for 'ready')." -ForegroundColor Cyan
Write-Host "Open http://localhost:5173" -ForegroundColor Green
npm --prefix web run dev
