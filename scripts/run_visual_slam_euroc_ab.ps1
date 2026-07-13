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

    # The demo interprets zero as all available frames (its omitted default is 400).
    [int]$MaxFrames = 0,

    [switch]$Resume,

    [switch]$ContinueOnError
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

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
    (Get-FileHash -LiteralPath $env:ORT_DYLIB_PATH -Algorithm SHA256).Hash
} else {
    $null
}

$commonArgs = @(
    '--max-frames', $MaxFrames.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--gravity', '0,0,-9.81',
    '--feature-extractor', 'superpoint-onnx',
    '--superpoint-onnx-model', $SuperPointModel,
    '--cross-check-matcher',
    '--keyframe-min-translation', '0.1',
    '--max-pose-jump-meters', '0.2',
    '--stereo-landmark-replenish',
    '--projection-guided-tracking',
    '--covisibility-local-map-max-keyframes', '10'
)

# Keep the baseline genuinely loop-free.  Pose-graph refinement consumes
# shared-landmark loop candidates even when appearance retrieval is disabled,
# so none of its switches belong in the common argument set.
$variantArguments = [ordered]@{
    no_loop = @()
    appearance_loop = @(
        '--pose-graph-refinement',
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

$experiment = [ordered]@{
    schema_version = 2
    created_at = (Get-Date).ToString('o')
    git_sha = $gitSha
    git_dirty = ($gitStatus.Count -gt 0)
    git_status_porcelain = $gitStatus
    git_diff_sha256 = $gitDiffSha256
    executable = $Executable
    executable_sha256 = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    dataset_root = $DatasetRoot
    superpoint_model = $SuperPointModel
    superpoint_model_sha256 = (Get-FileHash -LiteralPath $SuperPointModel -Algorithm SHA256).Hash
    repetitions = $Repetitions
    sequences = $Sequences
    max_frames = $MaxFrames
    protocol = [ordered]@{
        common_arguments = $commonArgs
        variant_arguments = $variantArguments
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
        powershell_version = $PSVersionTable.PSVersion.ToString()
        rustc_version = $rustcVersion
        cargo_version = $cargoVersion
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
        'repetitions',
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
                started_at = $startedAt.ToString('o')
                finished_at = $null
                elapsed_seconds = $null
                process_exit_code = $null
                exit_code = $null
                validation_error = $null
                sampled_peak_working_set_bytes = $null
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
                while (-not $process.WaitForExit(250)) {
                    $process.Refresh()
                    $peakWorkingSetBytes = [math]::Max(
                        $peakWorkingSetBytes,
                        $process.WorkingSet64
                    )
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
            $validationError = $null
            if ($processExitCode -eq 0) {
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
