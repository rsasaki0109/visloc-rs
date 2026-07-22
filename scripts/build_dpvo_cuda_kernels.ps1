param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repo "native\dpvo_cuda\dpvo_corr.cu"
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvcc)) {
    throw "nvcc not found at $nvcc"
}
$cl = Get-ChildItem `
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" `
    -Recurse -Filter cl.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -like "*Hostx64\x64\cl.exe" } |
    Sort-Object FullName -Descending |
    Select-Object -First 1
if (-not $cl) {
    throw "Visual Studio x64 cl.exe was not found"
}
$compilerDirectory = Split-Path -Parent $cl.FullName
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$output = Join-Path (Resolve-Path -LiteralPath $OutputDirectory) "visloc_dpvo_cuda.dll"
& $nvcc -O3 --use_fast_math -std=c++17 -shared -ccbin $compilerDirectory -Xcompiler "/MD" `
    -o $output $source
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}
Get-FileHash -Algorithm SHA256 -LiteralPath $output | Format-List
