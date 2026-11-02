# Run Python in the optional Windows AMD environment.
# Example: .\gpu.ps1 scripts/pipeline.py --dataset msrvtt --cached-only --device cuda
$ErrorActionPreference = 'Stop'
$previousHipDevice = $env:HIP_VISIBLE_DEVICES
$previousVirtualEnv = $env:VIRTUAL_ENV
$pythonArguments = $args
Push-Location $PSScriptRoot
try {
    # Explicit environment selection takes precedence over private local settings.
    $deviceFile = Join-Path $PSScriptRoot '.gpu-device'
    if (-not $env:HIP_VISIBLE_DEVICES -and (Test-Path -LiteralPath $deviceFile)) {
        $deviceSelection = (Get-Content -LiteralPath $deviceFile -Raw).Trim()
        if ($deviceSelection -notmatch '^\d+(,\d+)*$') {
            throw 'Invalid device selection in .gpu-device.'
        }
        $env:HIP_VISIBLE_DEVICES = $deviceSelection
    }
    # This launcher selects its own environment, independent of the active shell.
    $env:VIRTUAL_ENV = $null
    & uv run --project environments/amd --locked --offline python @pythonArguments
    $pythonExitCode = $LASTEXITCODE
} finally {
    $env:HIP_VISIBLE_DEVICES = $previousHipDevice
    $env:VIRTUAL_ENV = $previousVirtualEnv
    Pop-Location
}
exit $pythonExitCode
