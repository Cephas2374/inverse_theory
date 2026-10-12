# One-time install of the toolchain into this folder:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
. "$Root\activate.ps1"
New-Item -ItemType Directory -Force "$Root\tools" | Out-Null
$mm = "$Root\tools\micromamba.exe"
if (-not (Test-Path $mm)) {
    curl.exe -L --fail -o $mm https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64
}
& $mm create -y --rc-file "$Root\.condarc" --override-channels -c conda-forge --strict-channel-priority -r $env:MAMBA_ROOT_PREFIX -f "$Root\environment.yml"
if ($LASTEXITCODE) { throw "micromamba create failed" }
. "$Root\activate.ps1"
python -c "import simpeg, harmonica, networkx, torch, rasterio, geopandas; print('env OK')"
tectonic --version
