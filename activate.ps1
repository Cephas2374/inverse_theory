# Dot-source this in every new shell:  . .\activate.ps1
# Keeps ALL tools, caches and temp files inside this project folder (nothing on C:).
$Root = $PSScriptRoot
$env:MAMBA_ROOT_PREFIX   = "$Root\.mamba"
$env:CONDA_PKGS_DIRS     = "$Root\.cache\conda-pkgs"
$env:CONDARC             = "$Root\.condarc"
$env:MAMBARC             = "$Root\.condarc"
$env:PIP_CACHE_DIR       = "$Root\.cache\pip"
$env:TECTONIC_CACHE_DIR  = "$Root\.cache\tectonic"
$env:MPLCONFIGDIR        = "$Root\.cache\matplotlib"
$env:XDG_CACHE_HOME      = "$Root\.cache"
$env:HF_HOME             = "$Root\.cache\huggingface"
$env:TORCH_HOME          = "$Root\.cache\torch"
$env:POOCH_CACHE         = "$Root\.cache\pooch"
$env:ENSAIO_DATA_DIR     = "$Root\.cache\ensaio"
$env:JUPYTER_CONFIG_DIR  = "$Root\.cache\jupyter"
$env:IPYTHONDIR          = "$Root\.cache\ipython"
$env:TEMP                = "$Root\.cache\tmp"
$env:TMP                 = "$Root\.cache\tmp"
foreach ($d in $env:CONDA_PKGS_DIRS, $env:PIP_CACHE_DIR, $env:TECTONIC_CACHE_DIR, $env:MPLCONFIGDIR, $env:TEMP) {
    New-Item -ItemType Directory -Force $d | Out-Null
}
$Env = "$env:MAMBA_ROOT_PREFIX\envs\drc-deep"
if (Test-Path $Env) {
    $env:PATH = "$Env;$Env\Library\bin;$Env\Scripts;$Env\bin;$Root\tools;$env:PATH"
    $env:GDAL_DATA = "$Env\Library\share\gdal"
    $env:PROJ_DATA = "$Env\Library\share\proj"
    $env:GDAL_DRIVER_PATH = "$Env\Library\lib\gdalplugins"
}
