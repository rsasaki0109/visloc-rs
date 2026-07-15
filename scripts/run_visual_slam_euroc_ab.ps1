[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutRoot,

    [int]$Repetitions = 3,

    [ValidateSet('MH_01_easy', 'MH_03_medium', 'MH_05_difficult')]
    [string[]]$Sequences = @('MH_01_easy', 'MH_03_medium', 'MH_05_difficult'),

    [string]$Executable,

    [string]$SuperPointModel,

    # `cuda` is strict: model loading fails instead of silently falling back
    # to CPU when the CUDA execution provider or one of its DLLs is missing.
    [ValidateSet('cuda', 'cuda-then-cpu', 'cpu')]
    [string]$OnnxBackend = 'cuda',

    # The demo interprets zero as all available frames (its omitted default is 400).
    [int]$MaxFrames = 0,

    [double]$CandidateLoopEdgeWeight = 1.0,

    [switch]$CandidateLoopPoseInformation,

    [double]$CandidateLoopInformationMaxEigenvalue = 1.0,

    # Multiplicative strength of covariance-derived loop edges only. The
    # sequential PnP information matrices retain their calibrated scale.
    [double]$CandidateLoopInformationLoopEdgeScale = 1.0,

    [switch]$CandidateFuseLoopObservations,

    [switch]$CandidateLoopWeldingBa,

    [double]$MinAvailableMemoryGiB = 4.0,

    [double]$MinCommitHeadroomGiB = 4.0,

    [double]$MaxExternalCpuCores = 0.5,

    # Reject sustained external load while tolerating one-off scheduler or
    # monitoring spikes. Preflight remains a strict single-sample gate.
    [int]$ExternalCpuViolationSamples = 3,

    [int]$ResourceSampleIntervalSeconds = 5,

    [switch]$Resume,

    [switch]$ContinueOnError
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )

    $resolvedPath = [IO.Path]::GetFullPath($LiteralPath)
    $stream = [IO.File]::OpenRead($resolvedPath)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($sha256.ComputeHash($stream)).Replace('-', '')
    } finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

# Windows PowerShell evaluates parameter-default expressions before it
# populates `$PSScriptRoot`. Resolve repository-relative defaults in the script
# body so direct `powershell.exe -File ...` and PowerShell 7 behave alike.
if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $PSScriptRoot '..\target\release\examples\euroc_online_slam_vi_image_demo.exe'
}
if ([string]::IsNullOrWhiteSpace($SuperPointModel)) {
    $SuperPointModel = Join-Path $PSScriptRoot '..\models\superpoint_1500.onnx'
}

if ($Repetitions -lt 1) {
    throw 'Repetitions must be at least 1.'
}
if ($MaxFrames -lt 0) {
    throw 'MaxFrames must be zero (full sequence) or positive.'
}
if ([double]::IsNaN($CandidateLoopEdgeWeight) -or
    [double]::IsInfinity($CandidateLoopEdgeWeight) -or
    $CandidateLoopEdgeWeight -le 0.0) {
    throw 'CandidateLoopEdgeWeight must be finite and positive.'
}
if ([double]::IsNaN($CandidateLoopInformationMaxEigenvalue) -or
    [double]::IsInfinity($CandidateLoopInformationMaxEigenvalue) -or
    $CandidateLoopInformationMaxEigenvalue -le 0.0) {
    throw 'CandidateLoopInformationMaxEigenvalue must be finite and positive.'
}
if ([double]::IsNaN($CandidateLoopInformationLoopEdgeScale) -or
    [double]::IsInfinity($CandidateLoopInformationLoopEdgeScale) -or
    $CandidateLoopInformationLoopEdgeScale -le 0.0) {
    throw 'CandidateLoopInformationLoopEdgeScale must be finite and positive.'
}
if ($CandidateLoopWeldingBa -and -not $CandidateFuseLoopObservations) {
    throw 'CandidateLoopWeldingBa requires CandidateFuseLoopObservations.'
}
if ([double]::IsNaN($MinAvailableMemoryGiB) -or
    [double]::IsInfinity($MinAvailableMemoryGiB) -or
    $MinAvailableMemoryGiB -le 0.0) {
    throw 'MinAvailableMemoryGiB must be finite and positive.'
}
if ([double]::IsNaN($MinCommitHeadroomGiB) -or
    [double]::IsInfinity($MinCommitHeadroomGiB) -or
    $MinCommitHeadroomGiB -le 0.0) {
    throw 'MinCommitHeadroomGiB must be finite and positive.'
}
if ([double]::IsNaN($MaxExternalCpuCores) -or
    [double]::IsInfinity($MaxExternalCpuCores) -or
    $MaxExternalCpuCores -le 0.0) {
    throw 'MaxExternalCpuCores must be finite and positive.'
}
if ($ExternalCpuViolationSamples -lt 1) {
    throw 'ExternalCpuViolationSamples must be at least 1.'
}
if ($ResourceSampleIntervalSeconds -lt 1) {
    throw 'ResourceSampleIntervalSeconds must be at least 1.'
}

$minAvailableMemoryBytes = [int64]($MinAvailableMemoryGiB * 1GB)
$minCommitHeadroomBytes = [int64]($MinCommitHeadroomGiB * 1GB)

function Get-SystemMemorySnapshot {
    $memoryOperatingSystem = Get-CimInstance Win32_OperatingSystem
    [pscustomobject]@{
        available_physical_bytes = [int64]$memoryOperatingSystem.FreePhysicalMemory * 1KB
        commit_limit_bytes = [int64]$memoryOperatingSystem.TotalVirtualMemorySize * 1KB
        commit_headroom_bytes = [int64]$memoryOperatingSystem.FreeVirtualMemory * 1KB
    }
}

function Get-MemoryGateViolation {
    param(
        [Parameter(Mandatory = $true)]
        $Snapshot
    )

    $violations = [Collections.Generic.List[string]]::new()
    if ($Snapshot.available_physical_bytes -lt $minAvailableMemoryBytes) {
        $violations.Add(
            "available physical memory $($Snapshot.available_physical_bytes) bytes is below required $minAvailableMemoryBytes bytes"
        )
    }
    if ($Snapshot.commit_headroom_bytes -lt $minCommitHeadroomBytes) {
        $violations.Add(
            "commit headroom $($Snapshot.commit_headroom_bytes) bytes is below required $minCommitHeadroomBytes bytes"
        )
    }
    if ($violations.Count -eq 0) {
        return $null
    }
    return $violations -join '; '
}

function Get-ExternalProcessCpuSnapshot {
    param(
        [int[]]$ExcludedProcessIds = @()
    )

    $excluded = [Collections.Generic.HashSet[int]]::new()
    [void]$excluded.Add(0)
    foreach ($processId in $ExcludedProcessIds) {
        [void]$excluded.Add($processId)
    }
    $snapshot = @{}
    foreach ($candidate in Get-Process -ErrorAction SilentlyContinue) {
        if ($excluded.Contains($candidate.Id)) {
            continue
        }
        try {
            $cpuSeconds = $candidate.TotalProcessorTime.TotalSeconds
            if ([double]::IsNaN($cpuSeconds) -or [double]::IsInfinity($cpuSeconds)) {
                continue
            }
            $snapshot[$candidate.Id] = $cpuSeconds
        } catch {
            # A process may exit or deny access between enumeration and read.
        }
    }
    return $snapshot
}

function Get-ProcessAncestorIds {
    param(
        [Parameter(Mandatory = $true)]
        [int]$StartProcessId
    )

    $ids = [Collections.Generic.List[int]]::new()
    $seen = [Collections.Generic.HashSet[int]]::new()
    $currentId = $StartProcessId
    while ($currentId -gt 0 -and $seen.Add($currentId)) {
        $ids.Add($currentId)
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$currentId" `
            -ErrorAction SilentlyContinue
        if ($null -eq $processInfo) {
            break
        }
        $currentId = [int]$processInfo.ParentProcessId
    }
    return $ids.ToArray()
}

function Measure-ExternalCpuCores {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Previous,

        [Parameter(Mandatory = $true)]
        [hashtable]$Current,

        [Parameter(Mandatory = $true)]
        [double]$ElapsedSeconds
    )

    if ([double]::IsNaN($ElapsedSeconds) -or
        [double]::IsInfinity($ElapsedSeconds) -or
        $ElapsedSeconds -le 0.0) {
        return [double]::PositiveInfinity
    }
    $cpuSeconds = 0.0
    foreach ($processId in $Current.Keys) {
        if ($Previous.ContainsKey($processId)) {
            $delta = [double]$Current[$processId] - [double]$Previous[$processId]
            if ($delta -gt 0.0) {
                $cpuSeconds += $delta
            }
        }
    }
    return $cpuSeconds / $ElapsedSeconds
}

$DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot)
$OutRoot = [IO.Path]::GetFullPath($OutRoot)
$Executable = [IO.Path]::GetFullPath($Executable)
$SuperPointModel = [IO.Path]::GetFullPath($SuperPointModel)
$externalCpuExcludedProcessIds = @(
    Get-ProcessAncestorIds -StartProcessId $PID
)

foreach ($requiredPath in @($DatasetRoot, $Executable, $SuperPointModel)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}
if (-not $env:ORT_DYLIB_PATH -or -not (Test-Path -LiteralPath $env:ORT_DYLIB_PATH)) {
    throw 'ORT_DYLIB_PATH must point to the onnxruntime DLL used by the dynamic ONNX build.'
}

New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$gitSha = (& git -C $repositoryRoot rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0) {
    $gitSha = $null
}
$gitStatus = @(& git -C $repositoryRoot status --porcelain=v1 2>$null)
# Avoid checkout line-ending conversion warnings on stderr. Windows PowerShell
# promotes native stderr to a terminating error under ErrorActionPreference=Stop.
# This invocation only hashes the existing diff; disabling conversion does not
# modify the worktree.
$gitDiff = (& git -c core.autocrlf=false -C $repositoryRoot diff --binary HEAD 2>$null) -join "`n"
$sha256 = [Security.Cryptography.SHA256]::Create()
try {
    $gitDiffSha256 = if ($gitDiff.Length -gt 0) {
        [BitConverter]::ToString(
            $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($gitDiff))
        ).Replace('-', '')
    } else {
        $null
    }
} finally {
    $sha256.Dispose()
}
$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$operatingSystem = Get-CimInstance Win32_OperatingSystem
$rustcVersion = (& (Join-Path $HOME '.cargo\bin\rustc.exe') --version 2>$null)
$cargoVersion = (& (Join-Path $HOME '.cargo\bin\cargo.exe') --version 2>$null)
$ortDylibSha256 = if ($env:ORT_DYLIB_PATH -and (Test-Path -LiteralPath $env:ORT_DYLIB_PATH)) {
    Get-FileSha256 -LiteralPath $env:ORT_DYLIB_PATH
} else {
    $null
}

$commonArgs = @(
    '--max-frames', $MaxFrames.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--gravity', '0,0,-9.81',
    '--feature-extractor', 'superpoint-onnx',
    '--superpoint-onnx-model', $SuperPointModel,
    '--superpoint-onnx-backend', $OnnxBackend,
    '--cross-check-matcher',
    '--keyframe-min-translation', '0.1',
    '--max-pose-jump-meters', '0.2',
    '--stereo-landmark-replenish',
    '--projection-guided-tracking',
    '--covisibility-local-map-max-keyframes', '10'
)
$candidateLoopEdgeWeightText = $CandidateLoopEdgeWeight.ToString(
    'R',
    [Globalization.CultureInfo]::InvariantCulture
)
$candidateInformationArguments = if ($CandidateLoopPoseInformation) {
    @(
        '--pose-graph-refinement-loop-pose-information',
        '--pose-graph-refinement-loop-pose-information-max-eigenvalue',
        $CandidateLoopInformationMaxEigenvalue.ToString(
            'R',
            [Globalization.CultureInfo]::InvariantCulture
        ),
        '--pose-graph-refinement-loop-pose-information-loop-edge-scale',
        $CandidateLoopInformationLoopEdgeScale.ToString(
            'R',
            [Globalization.CultureInfo]::InvariantCulture
        )
    )
} else {
    @('--pose-graph-refinement-fixed-loop-edge-weight', $candidateLoopEdgeWeightText)
}
$candidateFusionArguments = if ($CandidateFuseLoopObservations) {
    $arguments = @('--pose-graph-refinement-fuse-loop-observations')
    if ($CandidateLoopWeldingBa) {
        $arguments += '--pose-graph-refinement-loop-welding-ba'
    }
    $arguments
} else {
    @()
}

# Keep the baseline genuinely loop-free.  Pose-graph refinement consumes
# shared-landmark loop candidates even when appearance retrieval is disabled,
# so none of its switches belong in the common argument set.
$variantArguments = [ordered]@{
    no_loop = @()
    appearance_loop = @(
        '--pose-graph-refinement'
    ) + $candidateInformationArguments + $candidateFusionArguments + @(
        '--pose-graph-refinement-verifier', 'pnp',
        '--pose-graph-refinement-gnc',
        '--pose-graph-refinement-pcm',
        '--pose-graph-refinement-pcm-pairwise-only',
        '--pose-graph-refinement-propagate',
        '--pose-graph-refinement-appearance-loops',
        '--pose-graph-refinement-appearance-confirmation-keyframes', '3',
        '--pose-graph-refinement-appearance-confirmation-max-misses', '2',
        '--pose-graph-refinement-appearance-projection-radius', '15',
        '--pose-graph-refinement-appearance-projection-min-matches', '50'
    )
}

$initialMemorySnapshot = Get-SystemMemorySnapshot

$experiment = [ordered]@{
    schema_version = 9
    created_at = (Get-Date).ToString('o')
    git_sha = $gitSha
    git_dirty = ($gitStatus.Count -gt 0)
    git_status_porcelain = $gitStatus
    git_diff_sha256 = $gitDiffSha256
    executable = $Executable
    executable_sha256 = Get-FileSha256 -LiteralPath $Executable
    dataset_root = $DatasetRoot
    superpoint_model = $SuperPointModel
    superpoint_model_sha256 = Get-FileSha256 -LiteralPath $SuperPointModel
    repetitions = $Repetitions
    sequences = $Sequences
    max_frames = $MaxFrames
    protocol = [ordered]@{
        common_arguments = $commonArgs
        onnx_backend = $OnnxBackend
        candidate_loop_pose_information = [bool]$CandidateLoopPoseInformation
        candidate_loop_information_max_eigenvalue = $CandidateLoopInformationMaxEigenvalue
        candidate_loop_information_loop_edge_scale = $CandidateLoopInformationLoopEdgeScale
        candidate_fuse_loop_observations = [bool]$CandidateFuseLoopObservations
        candidate_loop_welding_ba = [bool]$CandidateLoopWeldingBa
        variant_arguments = $variantArguments
        resource_gate = [ordered]@{
            minimum_available_physical_bytes = $minAvailableMemoryBytes
            minimum_commit_headroom_bytes = $minCommitHeadroomBytes
            maximum_external_cpu_cores = $MaxExternalCpuCores
            external_cpu_violation_samples = $ExternalCpuViolationSamples
            sample_interval_seconds = $ResourceSampleIntervalSeconds
        }
    }
    ort_dylib_path = $env:ORT_DYLIB_PATH
    ort_dylib_sha256 = $ortDylibSha256
    host = [ordered]@{
        computer_name = $env:COMPUTERNAME
        os_caption = $operatingSystem.Caption
        os_version = $operatingSystem.Version
        processor = $processor.Name
        logical_processors = $processor.NumberOfLogicalProcessors
        total_memory_bytes = [int64]$operatingSystem.TotalVisibleMemorySize * 1KB
        initial_available_physical_bytes = $initialMemorySnapshot.available_physical_bytes
        initial_commit_limit_bytes = $initialMemorySnapshot.commit_limit_bytes
        initial_commit_headroom_bytes = $initialMemorySnapshot.commit_headroom_bytes
        powershell_version = $PSVersionTable.PSVersion.ToString()
        rustc_version = $rustcVersion
        cargo_version = $cargoVersion
        external_cpu_excluded_process_ids = $externalCpuExcludedProcessIds
    }
    runs = @()
}

$experimentPath = Join-Path $OutRoot 'experiment_manifest.json'
if ($Resume -and (Test-Path -LiteralPath $experimentPath)) {
    $existingExperiment = Get-Content -LiteralPath $experimentPath -Raw |
        ConvertFrom-Json
    $resumeMismatches = [Collections.Generic.List[string]]::new()
    foreach ($field in @(
        'schema_version',
        'executable_sha256',
        'superpoint_model_sha256',
        'ort_dylib_sha256',
        'dataset_root',
        'max_frames'
    )) {
        if ($existingExperiment.$field -ne $experiment.$field) {
            $resumeMismatches.Add($field)
        }
    }
    $existingSequences = @($existingExperiment.sequences) -join "`n"
    $requestedSequences = @($experiment.sequences) -join "`n"
    if ($existingSequences -ne $requestedSequences) {
        $resumeMismatches.Add('sequences')
    }
    $existingProtocol = if ($existingExperiment.PSObject.Properties.Name -contains 'protocol') {
        $existingExperiment.protocol | ConvertTo-Json -Depth 8 -Compress
    } else {
        $null
    }
    $requestedProtocol = $experiment.protocol | ConvertTo-Json -Depth 8 -Compress
    if ($existingProtocol -ne $requestedProtocol) {
        $resumeMismatches.Add('protocol')
    }
    # A completed prefix may be extended with more repetitions, but shrinking
    # the declared matrix would silently orphan already-recorded runs.
    if ([int]$Repetitions -lt [int]$existingExperiment.repetitions) {
        $resumeMismatches.Add('repetitions_decrease')
    }
    if ($resumeMismatches.Count -gt 0) {
        throw "Resume configuration does not match the existing experiment: $($resumeMismatches -join ', ')"
    }
    $experiment.created_at = $existingExperiment.created_at
} else {
    $experiment | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $experimentPath -Encoding utf8
}

foreach ($sequence in $Sequences) {
    $datasetDir = Join-Path $DatasetRoot $sequence
    if (-not (Test-Path -LiteralPath (Join-Path $datasetDir 'mav0'))) {
        throw "EuRoC sequence is missing mav0: $datasetDir"
    }

    for ($repetition = 1; $repetition -le $Repetitions; $repetition++) {
        $variantOrder = if ($repetition % 2 -eq 1) {
            @('no_loop', 'appearance_loop')
        } else {
            @('appearance_loop', 'no_loop')
        }
        foreach ($variant in $variantOrder) {
            $runName = '{0}_{1}_r{2:d2}' -f $sequence, $variant, $repetition
            $runDir = Join-Path $OutRoot $runName
            if (Test-Path -LiteralPath $runDir) {
                $existingManifestPath = Join-Path $runDir 'run_manifest.json'
                $existingSummaryPath = Join-Path $runDir 'summary.txt'
                if ($Resume -and (Test-Path -LiteralPath $existingManifestPath)) {
                    $existingRun = Get-Content -LiteralPath $existingManifestPath -Raw |
                        ConvertFrom-Json
                    if ($existingRun.exit_code -eq 0 -and (Test-Path -LiteralPath $existingSummaryPath)) {
                        Write-Host "skipping completed $runName"
                        $experiment.runs += $existingRun
                        $experiment | ConvertTo-Json -Depth 8 |
                            Set-Content -LiteralPath $experimentPath -Encoding utf8
                        continue
                    }
                }
                throw "Refusing to overwrite incomplete or failed run directory: $runDir"
            }

            $runExecutableSha256 = Get-FileSha256 -LiteralPath $Executable
            $runSuperPointModelSha256 = Get-FileSha256 -LiteralPath $SuperPointModel
            $runOrtDylibSha256 = Get-FileSha256 -LiteralPath $env:ORT_DYLIB_PATH
            if ($runExecutableSha256 -ne $experiment.executable_sha256 -or
                $runSuperPointModelSha256 -ne $experiment.superpoint_model_sha256 -or
                $runOrtDylibSha256 -ne $experiment.ort_dylib_sha256) {
                throw "Artifact hash changed before ${runName}; refusing a mixed-binary matrix"
            }

            $preflightMemorySnapshot = Get-SystemMemorySnapshot
            $preflightMemoryViolation = Get-MemoryGateViolation -Snapshot $preflightMemorySnapshot
            if ($null -ne $preflightMemoryViolation) {
                throw "Resource preflight failed before ${runName}: $preflightMemoryViolation"
            }
            $preflightCpuStartedAt = Get-Date
            $preflightCpuStart = Get-ExternalProcessCpuSnapshot `
                -ExcludedProcessIds $externalCpuExcludedProcessIds
            Start-Sleep -Seconds 2
            $preflightCpuFinishedAt = Get-Date
            $preflightCpuEnd = Get-ExternalProcessCpuSnapshot `
                -ExcludedProcessIds $externalCpuExcludedProcessIds
            $preflightExternalCpuCores = Measure-ExternalCpuCores `
                -Previous $preflightCpuStart `
                -Current $preflightCpuEnd `
                -ElapsedSeconds ($preflightCpuFinishedAt - $preflightCpuStartedAt).TotalSeconds
            if ($preflightExternalCpuCores -gt $MaxExternalCpuCores) {
                throw "Resource preflight failed before ${runName}: external CPU load $([math]::Round($preflightExternalCpuCores, 3)) cores exceeds allowed $MaxExternalCpuCores cores"
            }
            New-Item -ItemType Directory -Path $runDir | Out-Null

            $arguments = @('--euroc-dir', $datasetDir, '--out-dir', $runDir) +
                $commonArgs + @($variantArguments[$variant])

            $startedAt = Get-Date
            $runRecord = [ordered]@{
                name = $runName
                sequence = $sequence
                variant = $variant
                repetition = $repetition
                output_dir = $runDir
                arguments = $arguments
                executable_sha256 = $runExecutableSha256
                superpoint_model_sha256 = $runSuperPointModelSha256
                ort_dylib_sha256 = $runOrtDylibSha256
                started_at = $startedAt.ToString('o')
                finished_at = $null
                elapsed_seconds = $null
                process_exit_code = $null
                exit_code = $null
                validation_error = $null
                sampled_peak_working_set_bytes = $null
                preflight_available_physical_bytes = $preflightMemorySnapshot.available_physical_bytes
                preflight_commit_limit_bytes = $preflightMemorySnapshot.commit_limit_bytes
                preflight_commit_headroom_bytes = $preflightMemorySnapshot.commit_headroom_bytes
                minimum_available_physical_bytes = $preflightMemorySnapshot.available_physical_bytes
                minimum_commit_headroom_bytes = $preflightMemorySnapshot.commit_headroom_bytes
                preflight_external_cpu_cores = $preflightExternalCpuCores
                sampled_max_external_cpu_cores = $preflightExternalCpuCores
                maximum_consecutive_external_cpu_violations = 0
                external_cpu_excluded_process_ids = $externalCpuExcludedProcessIds
            }
            $runRecord | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir 'run_manifest.json') -Encoding utf8

            Write-Host "[$($startedAt.ToString('s'))] starting $runName"
            $startInfo = [Diagnostics.ProcessStartInfo]::new()
            $startInfo.FileName = $Executable
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
            foreach ($argument in $arguments) {
                [void]$startInfo.ArgumentList.Add($argument)
            }
            $process = [Diagnostics.Process]::new()
            $process.StartInfo = $startInfo
            $stdoutStream = [IO.File]::Create((Join-Path $runDir 'stdout.log'))
            $stderrStream = [IO.File]::Create((Join-Path $runDir 'stderr.log'))
            try {
                if (-not $process.Start()) {
                    throw "Failed to start run: $runName"
                }
                $stdoutCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdoutStream)
                $stderrCopy = $process.StandardError.BaseStream.CopyToAsync($stderrStream)
                $peakWorkingSetBytes = 0L
                $minimumAvailablePhysicalBytes = $preflightMemorySnapshot.available_physical_bytes
                $minimumCommitHeadroomBytes = $preflightMemorySnapshot.commit_headroom_bytes
                $resourceValidationError = $null
                $maximumExternalCpuCores = $preflightExternalCpuCores
                $consecutiveExternalCpuViolations = 0
                $maximumConsecutiveExternalCpuViolations = 0
                $runCpuExcludedProcessIds = @(
                    $externalCpuExcludedProcessIds
                    $process.Id
                )
                $previousExternalCpuSnapshot = Get-ExternalProcessCpuSnapshot `
                    -ExcludedProcessIds $runCpuExcludedProcessIds
                $previousExternalCpuSampleAt = Get-Date
                $nextResourceSampleAt = (Get-Date).AddSeconds($ResourceSampleIntervalSeconds)
                while (-not $process.WaitForExit(250)) {
                    $process.Refresh()
                    $peakWorkingSetBytes = [math]::Max(
                        $peakWorkingSetBytes,
                        $process.WorkingSet64
                    )
                    if ((Get-Date) -ge $nextResourceSampleAt) {
                        $memorySnapshot = Get-SystemMemorySnapshot
                        $minimumAvailablePhysicalBytes = [math]::Min(
                            $minimumAvailablePhysicalBytes,
                            $memorySnapshot.available_physical_bytes
                        )
                        $minimumCommitHeadroomBytes = [math]::Min(
                            $minimumCommitHeadroomBytes,
                            $memorySnapshot.commit_headroom_bytes
                        )
                        $memoryViolation = Get-MemoryGateViolation -Snapshot $memorySnapshot
                        if ($null -ne $memoryViolation) {
                            $resourceValidationError = "resource gate failed during ${runName}: $memoryViolation"
                            $process.Kill()
                            [void]$process.WaitForExit()
                            break
                        }
                        $externalCpuSampleAt = Get-Date
                        $externalCpuSnapshot = Get-ExternalProcessCpuSnapshot `
                            -ExcludedProcessIds $runCpuExcludedProcessIds
                        $externalCpuCores = Measure-ExternalCpuCores `
                            -Previous $previousExternalCpuSnapshot `
                            -Current $externalCpuSnapshot `
                            -ElapsedSeconds ($externalCpuSampleAt - $previousExternalCpuSampleAt).TotalSeconds
                        $maximumExternalCpuCores = [math]::Max(
                            $maximumExternalCpuCores,
                            $externalCpuCores
                        )
                        if ($externalCpuCores -gt $MaxExternalCpuCores) {
                            $consecutiveExternalCpuViolations += 1
                            $maximumConsecutiveExternalCpuViolations = [math]::Max(
                                $maximumConsecutiveExternalCpuViolations,
                                $consecutiveExternalCpuViolations
                            )
                            if ($consecutiveExternalCpuViolations -ge $ExternalCpuViolationSamples) {
                                $resourceValidationError = "resource gate failed during ${runName}: external CPU load $([math]::Round($externalCpuCores, 3)) cores exceeded allowed $MaxExternalCpuCores cores for $consecutiveExternalCpuViolations consecutive samples"
                                $process.Kill()
                                [void]$process.WaitForExit()
                                break
                            }
                        } else {
                            $consecutiveExternalCpuViolations = 0
                        }
                        $previousExternalCpuSnapshot = $externalCpuSnapshot
                        $previousExternalCpuSampleAt = $externalCpuSampleAt
                        $nextResourceSampleAt = (Get-Date).AddSeconds($ResourceSampleIntervalSeconds)
                    }
                }
                [void]$stdoutCopy.GetAwaiter().GetResult()
                [void]$stderrCopy.GetAwaiter().GetResult()
                $exitCode = $process.ExitCode
            } finally {
                $stdoutStream.Dispose()
                $stderrStream.Dispose()
                $process.Dispose()
            }
            $finishedAt = Get-Date

            $processExitCode = $exitCode
            $validationError = $resourceValidationError
            if ($null -eq $validationError -and $processExitCode -eq 0) {
                $summaryPath = Join-Path $runDir 'summary.txt'
                if (-not (Test-Path -LiteralPath $summaryPath)) {
                    $validationError = 'successful process did not produce summary.txt'
                } else {
                    $summaryText = Get-Content -LiteralPath $summaryPath -Raw
                    $poseGraphMatch = [regex]::Match(
                        $summaryText,
                        '(?m)^pose_graph_refinement=(true|false)\r?$'
                    )
                    $expectedPoseGraph = $variant -eq 'appearance_loop'
                    if (-not $poseGraphMatch.Success) {
                        $validationError = 'summary.txt does not record pose_graph_refinement'
                    } else {
                        $actualPoseGraph = $poseGraphMatch.Groups[1].Value -eq 'true'
                        if ($actualPoseGraph -ne $expectedPoseGraph) {
                            $validationError = "pose_graph_refinement=$actualPoseGraph, expected $expectedPoseGraph for $variant"
                        }
                    }
                    if ($null -eq $validationError) {
                        $fixedWeightMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_fixed_loop_edge_weight=(None|Some\(([^)]+)\))\r?$'
                        )
                        if (-not $fixedWeightMatch.Success) {
                            $validationError = 'summary.txt does not record the fixed loop-edge weight'
                        } else {
                            $actualFixedWeight = if ($fixedWeightMatch.Groups[1].Value -eq 'None') {
                                $null
                            } else {
                                [double]::Parse(
                                    $fixedWeightMatch.Groups[2].Value,
                                    [Globalization.CultureInfo]::InvariantCulture
                                )
                            }
                            $expectedFixedWeight = if (
                                $variant -eq 'appearance_loop' -and
                                -not $CandidateLoopPoseInformation
                            ) {
                                $CandidateLoopEdgeWeight
                            } else {
                                $null
                            }
                            $fixedWeightMatches = if ($null -eq $expectedFixedWeight) {
                                $null -eq $actualFixedWeight
                            } else {
                                $null -ne $actualFixedWeight -and
                                    [math]::Abs($actualFixedWeight - $expectedFixedWeight) -le 1.0e-12
                            }
                            if (-not $fixedWeightMatches) {
                                $validationError = "fixed loop-edge weight mismatch for ${variant}: actual=$actualFixedWeight expected=$expectedFixedWeight"
                            }
                        }
                    }
                    if ($null -eq $validationError) {
                        $informationMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_loop_pose_information=(true|false)\r?$'
                        )
                        if (-not $informationMatch.Success) {
                            $validationError = 'summary.txt does not record loop pose information mode'
                        } else {
                            $actualInformation = $informationMatch.Groups[1].Value -eq 'true'
                            $expectedInformation =
                                $variant -eq 'appearance_loop' -and
                                [bool]$CandidateLoopPoseInformation
                            if ($actualInformation -ne $expectedInformation) {
                                $validationError = "loop pose information mismatch for ${variant}: actual=$actualInformation expected=$expectedInformation"
                            }
                        }
                    }
                    if ($null -eq $validationError) {
                        $informationCapMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_loop_pose_information_max_eigenvalue=([^\r\n]+)\r?$'
                        )
                        if (-not $informationCapMatch.Success) {
                            $validationError = 'summary.txt does not record loop pose information spectral cap'
                        } else {
                            $actualInformationCap = [double]::Parse(
                                $informationCapMatch.Groups[1].Value,
                                [Globalization.CultureInfo]::InvariantCulture
                            )
                            $expectedInformationCap = if (
                                $variant -eq 'appearance_loop' -and
                                [bool]$CandidateLoopPoseInformation
                            ) {
                                $CandidateLoopInformationMaxEigenvalue
                            } else {
                                1.0
                            }
                            if ([math]::Abs($actualInformationCap - $expectedInformationCap) -gt
                                1.0e-9 * [math]::Max(1.0, [math]::Abs($expectedInformationCap))) {
                                $validationError = "loop pose information cap mismatch for ${variant}: actual=$actualInformationCap expected=$expectedInformationCap"
                            }
                        }
                    }
                    if ($null -eq $validationError) {
                        $informationScaleMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_loop_pose_information_loop_edge_scale=([^\r\n]+)\r?$'
                        )
                        if (-not $informationScaleMatch.Success) {
                            $validationError = 'summary.txt does not record loop-only pose information scale'
                        } else {
                            $actualInformationScale = [double]::Parse(
                                $informationScaleMatch.Groups[1].Value,
                                [Globalization.CultureInfo]::InvariantCulture
                            )
                            $expectedInformationScale = if (
                                $variant -eq 'appearance_loop' -and
                                [bool]$CandidateLoopPoseInformation
                            ) {
                                $CandidateLoopInformationLoopEdgeScale
                            } else {
                                1.0
                            }
                            if ([math]::Abs($actualInformationScale - $expectedInformationScale) -gt
                                1.0e-9 * [math]::Max(1.0, [math]::Abs($expectedInformationScale))) {
                                $validationError = "loop pose information scale mismatch for ${variant}: actual=$actualInformationScale expected=$expectedInformationScale"
                            }
                        }
                    }
                    if ($null -eq $validationError) {
                        $fusionMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_fuse_loop_observations=(true|false)\r?$'
                        )
                        if (-not $fusionMatch.Success) {
                            $validationError = 'summary.txt does not record loop observation fusion mode'
                        } else {
                            $actualFusion = $fusionMatch.Groups[1].Value -eq 'true'
                            $expectedFusion =
                                $variant -eq 'appearance_loop' -and
                                [bool]$CandidateFuseLoopObservations
                            if ($actualFusion -ne $expectedFusion) {
                                $validationError = "loop observation fusion mismatch for ${variant}: actual=$actualFusion expected=$expectedFusion"
                            }
                        }
                    }
                    if ($null -eq $validationError) {
                        $weldingMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_loop_welding_ba=(true|false)\r?$'
                        )
                        if (-not $weldingMatch.Success) {
                            $validationError = 'summary.txt does not record loop welding BA mode'
                        } else {
                            $actualWelding = $weldingMatch.Groups[1].Value -eq 'true'
                            $expectedWelding =
                                $variant -eq 'appearance_loop' -and
                                [bool]$CandidateLoopWeldingBa
                            if ($actualWelding -ne $expectedWelding) {
                                $validationError = "loop welding BA mismatch for ${variant}: actual=$actualWelding expected=$expectedWelding"
                            }
                        }
                    }
                    if ($null -eq $validationError) {
                        $pcmMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_pcm=(true|false)\r?$'
                        )
                        $individualMatch = [regex]::Match(
                            $summaryText,
                            '(?m)^pose_graph_refinement_pcm_require_individual=(true|false)\r?$'
                        )
                        if (-not $pcmMatch.Success -or -not $individualMatch.Success) {
                            $validationError = 'summary.txt does not record the complete PCM configuration'
                        } else {
                            $actualPcm = $pcmMatch.Groups[1].Value -eq 'true'
                            $actualRequireIndividual = $individualMatch.Groups[1].Value -eq 'true'
                            $expectedPcm = $variant -eq 'appearance_loop'
                            $expectedRequireIndividual = $variant -ne 'appearance_loop'
                            if ($actualPcm -ne $expectedPcm -or
                                $actualRequireIndividual -ne $expectedRequireIndividual) {
                                $validationError = "PCM configuration mismatch for ${variant}: pcm=$actualPcm require_individual=$actualRequireIndividual"
                            }
                        }
                    }
                }
            }
            if ($null -ne $validationError) {
                # Reserve a runner-side non-zero status while retaining the
                # binary's real exit code separately in the manifest.
                $exitCode = 3
            }

            $runRecord.finished_at = $finishedAt.ToString('o')
            $runRecord.elapsed_seconds = [math]::Round(($finishedAt - $startedAt).TotalSeconds, 3)
            $runRecord.process_exit_code = $processExitCode
            $runRecord.exit_code = $exitCode
            $runRecord.validation_error = $validationError
            $runRecord.sampled_peak_working_set_bytes = $peakWorkingSetBytes
            $runRecord.minimum_available_physical_bytes = $minimumAvailablePhysicalBytes
            $runRecord.minimum_commit_headroom_bytes = $minimumCommitHeadroomBytes
            $runRecord.sampled_max_external_cpu_cores = $maximumExternalCpuCores
            $runRecord.maximum_consecutive_external_cpu_violations =
                $maximumConsecutiveExternalCpuViolations
            $runRecord | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir 'run_manifest.json') -Encoding utf8
            $experiment.runs += $runRecord
            $experiment | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $experimentPath -Encoding utf8

            Write-Host "[$($finishedAt.ToString('s'))] finished $runName exit=$exitCode"
            if ($exitCode -ne 0 -and -not $ContinueOnError) {
                throw "Run failed with exit code ${exitCode}: $runName"
            }
        }
    }
}
