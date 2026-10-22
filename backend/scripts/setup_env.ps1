# Create the Python 3.11 virtual environment at the repository root (.venv)
# and install the backend dependencies.
#
#   powershell -ExecutionPolicy Bypass -File backend\scripts\setup_env.ps1          # CUDA 12.6
#   powershell -ExecutionPolicy Bypass -File backend\scripts\setup_env.ps1 -Cpu     # CPU only
param([switch]$Cpu)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    py -3.11 -m venv $venv
}

# Unpack wheels next to the venv, not in %TEMP%: the CUDA torch wheel needs
# several GB of scratch space and the system drive may not have it.
$tmp = Join-Path $venv ".pip-tmp"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp

& $py -m pip install --upgrade pip

$index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu126" }
& $py -m pip install --no-cache-dir torch torchvision --index-url $index
& $py -m pip install --no-cache-dir -r (Join-Path $root "backend\requirements.txt")

Remove-Item -Recurse -Force $tmp

& $py -c "import sys, torch; print(sys.version); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
