# Creates the Python 3.12 venv with CUDA torch, installs the vendored libraries
# and the server package, enables the gitleaks hooks, and installs web deps.
# Run from anywhere: .\scripts\setup_venv.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Step([string]$Name, [scriptblock]$Block) {
    Write-Host "==> $Name" -ForegroundColor Cyan
    # Native tools (uv, npm) write progress to stderr. Under "Stop", Windows
    # PowerShell 5.1 turns that into a terminating error, so judge success by
    # the exit code instead.
    $ErrorActionPreference = "Continue"
    & $Block 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "Step failed ($code): $Name" }
}

if (-not (Test-Path .venv\Scripts\python.exe)) {
    # --clear also repairs a venv left half-built by an interrupted run.
    Invoke-Step "Create Python 3.12 venv" { uv venv -p 3.12 --clear .venv }
}
Invoke-Step "Install CUDA torch" { uv pip install --python .venv -r RealtimeSTT/requirements-gpu-torch.txt }
Invoke-Step "Install RealtimeSTT[faster-whisper], RealtimeTTS[kokoro], server" {
    uv pip install --python .venv -e "RealtimeSTT[faster-whisper]" -e "RealtimeTTS[kokoro]" -e "server[dev,transformers]"
}
# A dependency may have pulled in CPU torch; reinstall the CUDA build over it.
Invoke-Step "Re-assert CUDA torch" { uv pip install --python .venv -r RealtimeSTT/requirements-gpu-torch.txt }
Invoke-Step "Check CUDA" {
    .venv\Scripts\python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch not active'; print(torch.__version__, torch.cuda.get_device_name(0))"
}
Invoke-Step "Enable gitleaks hooks" { git config core.hooksPath .githooks }
if (Test-Path web/package.json) {
    Invoke-Step "Install web dependencies" { npm --prefix web install }
}
Write-Host "Setup complete." -ForegroundColor Green
