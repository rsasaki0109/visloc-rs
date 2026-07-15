[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DatasetRoot,
    [Parameter(Mandatory = $true)][string]$OutRoot,
    [int]$RepetitionStart = 1,
    [int]$Repetitions = 3,
    [ValidateSet('MH_01_easy', 'MH_03_medium', 'MH_05_difficult')]
    [string[]]$Sequences = @('MH_01_easy', 'MH_03_medium', 'MH_05_difficult'),
    [string]$Executable,
    [string]$SuperPointModel,
    [ValidateSet('cuda', 'cuda-then-cpu', 'cpu')][string]$OnnxBackend = 'cuda',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    $stream = [IO.File]::OpenRead([IO.Path]::GetFullPath($LiteralPath))
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        [BitConverter]::ToString($sha256.ComputeHash($stream)).Replace('-', '')
    } finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Read-Summary {
    param([Parameter(Mandatory = $true)][string]$LiteralPath)
    $summary = @{}
    foreach ($line in Get-Content -LiteralPath $LiteralPath) {
        if ($line -match '^([^=]+)=(.*)$') {
            $summary[$matches[1]] = $matches[2]
        }
    }
    $summary
}

function Write-JsonAtomic {
    param([Parameter(Mandatory = $true)]$Value,
          [Parameter(Mandatory = $true)][string]$LiteralPath)
    $temporaryPath = "$LiteralPath.tmp"
    $Value | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath $temporaryPath -Encoding utf8
    Move-Item -LiteralPath $temporaryPath -Destination $LiteralPath -Force
}

if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $PSScriptRoot '..\target\release\examples\euroc_online_slam_vi_image_demo.exe'
}
if ([string]::IsNullOrWhiteSpace($SuperPointModel)) {
    $SuperPointModel = Join-Path $PSScriptRoot '..\models\superpoint_1500.onnx'
}
if ($RepetitionStart -lt 1 -or $Repetitions -lt 1) {
    throw 'RepetitionStart and Repetitions must be at least 1.'
}

$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$OutRoot = [IO.Path]::GetFullPath($OutRoot)
$Executable = [IO.Path]::GetFullPath($Executable)
$SuperPointModel = [IO.Path]::GetFullPath($SuperPointModel)
foreach ($requiredPath in @($DatasetRoot, $Executable, $SuperPointModel)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}
if (-not $env:ORT_DYLIB_PATH -or -not (Test-Path -LiteralPath $env:ORT_DYLIB_PATH)) {
    throw 'ORT_DYLIB_PATH must point to the ONNX Runtime DLL.'
}
New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null

$executableSha256 = Get-FileSha256 $Executable
$modelSha256 = Get-FileSha256 $SuperPointModel
$ortSha256 = Get-FileSha256 $env:ORT_DYLIB_PATH
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitSha = (& git -C $repositoryRoot rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0) { $gitSha = $null }
$gitStatus = @(& git -C $repositoryRoot status --porcelain=v1 2>$null)

$commonArguments = @(
    '--max-frames','0','--gravity','0,0,-9.81',
    '--feature-extractor','superpoint-onnx','--superpoint-onnx-model',$SuperPointModel,
    '--superpoint-onnx-backend',$OnnxBackend,'--cross-check-matcher',
    '--keyframe-min-translation','0.1','--max-pose-jump-meters','0.2',
    '--stereo-landmark-replenish','--projection-guided-tracking',
    '--covisibility-local-map-max-keyframes','10',
    '--pose-graph-refinement','--pose-graph-refinement-fixed-loop-edge-weight','0.1',
    '--pose-graph-refinement-verifier','pnp','--pose-graph-refinement-gnc',
    '--pose-graph-refinement-pcm','--pose-graph-refinement-pcm-pairwise-only',
    '--pose-graph-refinement-propagate','--pose-graph-refinement-appearance-loops',
    '--pose-graph-refinement-appearance-confirmation-keyframes','3',
    '--pose-graph-refinement-appearance-confirmation-max-misses','2',
    '--pose-graph-refinement-appearance-projection-radius','15',
    '--pose-graph-refinement-appearance-projection-min-matches','50',
    '--pose-graph-refinement-fuse-loop-observations',
    '--pose-graph-refinement-loop-welding-ba'
)

$candidateArguments = @(
    '--pose-prior-visual-override','--motion-vi-init','--motion-vi-init-after-static-give-up',
    '--motion-vi-init-max-velocity','10','--motion-vi-init-max-gyro-bias','0.2',
    '--motion-vi-init-max-accel-bias','1','--motion-vi-init-max-imu-nis-per-dof','20000',
    '--motion-vi-init-max-rotation-residual-rms-rad','0.01',
    '--motion-vi-init-max-velocity-residual-rms-mps','0.25',
    '--motion-vi-init-max-position-residual-rms-m','0.08','--local-vi-ba',
    '--local-vi-ba-marginalization','--local-vi-ba-initial-prior-std-devs','1,0.02,0.1',
    '--local-vi-ba-freeze-biases-above','0.9','--local-vi-ba-reject-writeback-above','1',
    '--local-vi-ba-reject-final-imu-nis-per-dof-above','5',
    '--local-vi-ba-reject-velocity-above','10',
    '--local-vi-ba-reject-pose-translation-above','0.2',
    '--local-vi-ba-reject-pose-rotation-above-deg','10',
    '--local-vi-ba-adaptive-velocity-gate','--covisibility-local-ba',
    '--covisibility-local-ba-max-seed-landmarks-for-activation','900',
    '--covisibility-local-ba-min-active-observations','30',
    '--covisibility-local-ba-general-stereo'
)

$protocol = [ordered]@{
    common_arguments = $commonArguments
    variant_arguments = [ordered]@{ control = @(); candidate = $candidateArguments }
    counterbalance = 'control-first when (repetition + sequence_index) is even; candidate-first otherwise'
}
$manifestPath = Join-Path $OutRoot 'tracking_cliff_tight_vi_manifest.json'
$manifest = [ordered]@{
    schema_version = 1; created_at = (Get-Date).ToString('o')
    git_sha = $gitSha; git_dirty = ($gitStatus.Count -gt 0); git_status_porcelain = $gitStatus
    executable = $Executable; executable_sha256 = $executableSha256
    superpoint_model = $SuperPointModel; superpoint_model_sha256 = $modelSha256
    ort_dylib_path = [IO.Path]::GetFullPath($env:ORT_DYLIB_PATH); ort_dylib_sha256 = $ortSha256
    dataset_root = $DatasetRoot; repetition_start = $RepetitionStart
    repetitions = $Repetitions; sequences = $Sequences; protocol = $protocol; runs = @()
}
if ($Resume -and (Test-Path -LiteralPath $manifestPath)) {
    $existing = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($field in @('executable_sha256','superpoint_model_sha256','ort_dylib_sha256','dataset_root')) {
        if ($existing.$field -ne $manifest.$field) { throw "Resume manifest mismatch: $field" }
    }
    if (($existing.protocol | ConvertTo-Json -Depth 12 -Compress) -ne
        ($manifest.protocol | ConvertTo-Json -Depth 12 -Compress)) {
        throw 'Resume manifest mismatch: protocol'
    }
    $manifest.created_at = $existing.created_at
}
Write-JsonAtomic $manifest $manifestPath

$repetitionEnd = $RepetitionStart + $Repetitions - 1
for ($repetition = $RepetitionStart; $repetition -le $repetitionEnd; $repetition++) {
    for ($sequenceIndex = 0; $sequenceIndex -lt $Sequences.Count; $sequenceIndex++) {
        $sequence = $Sequences[$sequenceIndex]
        $datasetDir = [IO.Path]::GetFullPath((Join-Path $DatasetRoot $sequence))
        if (-not (Test-Path -LiteralPath (Join-Path $datasetDir 'mav0'))) {
            throw "EuRoC sequence is missing mav0: $datasetDir"
        }
        $variantOrder = if (($repetition + $sequenceIndex) % 2 -eq 0) {
            @('control', 'candidate')
        } else {
            @('candidate', 'control')
        }

        foreach ($variant in $variantOrder) {
            $runName = '{0}_{1}_r{2:d2}' -f $sequence, $variant, $repetition
            $runDir = Join-Path $OutRoot $runName
            $summaryPath = Join-Path $runDir 'summary.txt'
            $runManifestPath = Join-Path $runDir 'run_manifest.json'
            $arguments = @('--euroc-dir',$datasetDir,'--out-dir',$runDir) + $commonArguments
            if ($variant -eq 'candidate') { $arguments += $candidateArguments }

            if (Test-Path -LiteralPath $runDir) {
                if (-not $Resume -or -not (Test-Path -LiteralPath $summaryPath)) {
                    throw "Refusing to overwrite incomplete or unapproved run directory: $runDir"
                }
                $summary = Read-Summary $summaryPath
                if ($summary.euroc_dir -ne $datasetDir -or $summary.frames_recorded -eq '0') {
                    throw "Existing run failed validation: $runName"
                }
                if ($variant -eq 'candidate') {
                    if ($summary.pose_prior_visual_override -ne 'true' -or
                        $summary.covisibility_local_ba_max_seed_landmarks_for_activation -ne 'Some(900)') {
                        throw "Existing candidate protocol mismatch: $runName"
                    }
                } elseif ($summary.pose_prior_visual_override -ne 'false' -or
                    $summary.covisibility_local_ba_enabled -ne 'false') {
                    throw "Existing control protocol mismatch: $runName"
                }
                $runRecord = [ordered]@{
                    run_name = $runName; sequence = $sequence; repetition = $repetition
                    variant = $variant; reused = $true; output_dir = $runDir
                    summary_sha256 = Get-FileSha256 $summaryPath
                    arguments = $arguments; exit_code = 0
                }
                Write-JsonAtomic $runRecord $runManifestPath
                $manifest.runs += $runRecord
                Write-JsonAtomic $manifest $manifestPath
                Write-Host "validated existing $runName"
                continue
            }

            foreach ($artifact in @(
                @($Executable,$executableSha256),
                @($SuperPointModel,$modelSha256),
                @($env:ORT_DYLIB_PATH,$ortSha256)
            )) {
                if ((Get-FileSha256 $artifact[0]) -ne $artifact[1]) {
                    throw "Artifact hash changed before ${runName}: $($artifact[0])"
                }
            }

            $startedAt = Get-Date
            Write-Host "starting $runName at $($startedAt.ToString('o'))"
            & $Executable @arguments
            $exitCode = $LASTEXITCODE
            $finishedAt = Get-Date
            $runRecord = [ordered]@{
                run_name = $runName; sequence = $sequence; repetition = $repetition
                variant = $variant; reused = $false; output_dir = $runDir
                started_at = $startedAt.ToString('o'); finished_at = $finishedAt.ToString('o')
                elapsed_seconds = ($finishedAt - $startedAt).TotalSeconds
                executable_sha256 = $executableSha256
                superpoint_model_sha256 = $modelSha256; ort_dylib_sha256 = $ortSha256
                arguments = $arguments; exit_code = $exitCode
                summary_sha256 = if (Test-Path -LiteralPath $summaryPath) {
                    Get-FileSha256 $summaryPath
                } else { $null }
            }
            New-Item -ItemType Directory -Path $runDir -Force | Out-Null
            Write-JsonAtomic $runRecord $runManifestPath
            $manifest.runs += $runRecord
            Write-JsonAtomic $manifest $manifestPath
            if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $summaryPath)) {
                throw "$runName failed with exit code $exitCode"
            }
            Write-Host "completed $runName in $([math]::Round(($finishedAt - $startedAt).TotalSeconds, 1)) s"
        }
    }
}

Write-Host "matrix complete: $manifestPath"
