$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$cache = Join-Path $root 'cache/verus'
$verusDir = Join-Path $cache 'verus-x86-win'
$verus = Join-Path $verusDir 'verus.exe'
if (!(Test-Path $verus)) { throw 'Run tools/verus/setup.ps1 first' }

function Update-SourceOnce([string] $source, [string] $before, [string] $after) {
    if (($source.Split(@($before), [StringSplitOptions]::None).Length - 1) -ne 1) {
        throw "Expected exactly one source anchor: $before"
    }
    return $source.Replace($before, $after)
}

$sourcePath = Join-Path $root 'fcs-core/src/cpu/output.rs'
$source = [IO.File]::ReadAllText($sourcePath).Replace("`r`n", "`n")
$sourcePathForRust = $sourcePath.Replace('\', '/')
$wrapperPath = Join-Path $cache 'output-wrapper.rs'
$mutantPath = Join-Path $cache 'output-mutant.rs'
$mutantWrapperPath = Join-Path $cache 'output-mutant-wrapper.rs'
$log = Join-Path $cache 'mutation.log'
$mutantPathForRust = $mutantPath.Replace('\', '/')
$wrapper = @"
mod cpu {
    #[path = "$sourcePathForRust"]
    mod output;
}
"@
[IO.File]::WriteAllText($wrapperPath, $wrapper)

$previousRustupHome = $env:RUSTUP_HOME
$previousToolchain = $env:RUSTUP_TOOLCHAIN
try {
    $env:RUSTUP_HOME = Join-Path $cache 'rustup'
    $env:RUSTUP_TOOLCHAIN = '1.99.1-x86_64-pc-windows-msvc'
    & $verus --crate-type lib --no-cheating $wrapperPath
    if ($LASTEXITCODE -ne 0) { throw 'Production output-row verification failed' }

    $mutantWrapper = @"
mod cpu {
    #[path = "$mutantPathForRust"]
    mod output;
}
"@
    [IO.File]::WriteAllText($mutantWrapperPath, $mutantWrapper)
    # Each broken writer must fail verification, not merely produce a Rust syntax error.
    $mutants = @(
        @('        x += 1;', '        x += 2;', 'upsampled row that skips output slots'),
        @('e += 1;', 'e += 2;', 'indexed writer that skips output slots'),
        @('ox += 4;', 'ox += 5;', 'tiled row that skips output slots'),
        @('while 4 <= hi - ox', 'while ox < hi', 'tiled row that calls quad outside the interior')
    )
    foreach ($m in $mutants) {
        [IO.File]::WriteAllText($mutantPath, (Update-SourceOnce $source $m[0] $m[1]))
        $ErrorActionPreference = 'Continue'
        & $verus --crate-type lib --no-cheating $mutantWrapperPath *> $log
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -eq 0) { throw "Verus accepted a $($m[2])" }
        if (!(Select-String -Path $log -Pattern 'invariant not satisfied|postcondition not satisfied|callee\.requires' -Quiet)) {
            Get-Content $log
            throw "Mutation failed for an unexpected reason: $($m[2])"
        }
        Write-Output "Verus rejected the $($m[2])"
    }
} finally {
    $env:RUSTUP_HOME = $previousRustupHome
    $env:RUSTUP_TOOLCHAIN = $previousToolchain
}
# The expected mutant failure must not become the calling CI shell's exit code.
exit 0
