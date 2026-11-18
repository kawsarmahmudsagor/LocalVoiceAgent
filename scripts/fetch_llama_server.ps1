# Downloads the latest official llama.cpp Windows CUDA 12 build (llama-server.exe
# plus the matching CUDA runtime DLLs) into tools/llama/, replacing what is there.
# Run from anywhere: .\scripts\fetch_llama_server.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
Set-Location (Split-Path -Parent $PSScriptRoot)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# "releases/latest" is not used: llama.cpp publishes its binary builds (tags b####)
# as prereleases, and "latest" can point at a release with no binaries.
$releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20" `
    -Headers @{ "User-Agent" = "fetch-llama-server" }

function Get-CudaVersion([string]$Name) {
    if ($Name -match "cuda-(12\.\d+)-x64\.zip$") { return [version]$Matches[1] }
    return $null
}

# Newest release that has a CUDA 12.x binaries + runtime pair; highest 12.x within it.
$release = $null; $pair = $null
foreach ($r in $releases) {
    $bins = @($r.assets | Where-Object { $_.name -like "llama-*-bin-win-cuda-12.*-x64.zip" })
    $runtimes = @($r.assets | Where-Object { $_.name -like "cudart-llama-bin-win-cuda-12.*-x64.zip" })
    $pair = $bins | ForEach-Object {
        $ver = Get-CudaVersion $_.name
        $rt = $runtimes | Where-Object { (Get-CudaVersion $_.name) -eq $ver } | Select-Object -First 1
        if ($ver -and $rt) { [pscustomobject]@{ Version = $ver; Bin = $_; Runtime = $rt } }
    } | Sort-Object Version -Descending | Select-Object -First 1
    if ($pair) { $release = $r; break }
}

if (-not $pair) {
    Write-Error "No llama-*-bin-win-cuda-12.*-x64.zip + cudart zip pair in the 20 newest llama.cpp releases."
    exit 1
}

$dest = Join-Path (Get-Location) "tools\llama"
$tmp = Join-Path $env:TEMP "llama-fetch-$($release.tag_name)"
New-Item -ItemType Directory -Force $tmp | Out-Null

foreach ($asset in @($pair.Bin, $pair.Runtime)) {
    $zip = Join-Path $tmp $asset.name
    Write-Host "Downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)"
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers @{ "User-Agent" = "fetch-llama-server" }
}

if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Force $dest | Out-Null
foreach ($asset in @($pair.Bin, $pair.Runtime)) {
    Expand-Archive -Path (Join-Path $tmp $asset.name) -DestinationPath $dest -Force
}
Remove-Item -Recurse -Force $tmp

# Some releases nest the files in a folder; make sure llama-server.exe is at the top.
$exe = Get-ChildItem -Path $dest -Recurse -Filter llama-server.exe | Select-Object -First 1
if (-not $exe) { Write-Error "llama-server.exe not found in the downloaded archive."; exit 1 }
if ($exe.DirectoryName -ne $dest) {
    Get-ChildItem -Path $exe.DirectoryName | Move-Item -Destination $dest -Force
}

Write-Host "Installed llama.cpp $($release.tag_name) (CUDA $($pair.Version)) into tools\llama" -ForegroundColor Green
