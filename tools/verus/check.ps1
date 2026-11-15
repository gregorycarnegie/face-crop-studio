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

# Production files Verus checks, each through a module wrapper of its own.
$files = @{
    output = @('fcs-core/src/cpu/output.rs', 'cpu', 'output')
    unpack = @('fcs-utils/src/image_utils/unpack.rs', 'image_utils', 'unpack')
}
$log = Join-Path $cache 'mutation.log'
function Write-Wrapper([string] $name, [string] $modulePath) {
    $parent, $module = $files[$name][1], $files[$name][2]
    $wrapperPath = Join-Path $cache "$name-wrapper.rs"
    $wrapper = @"
mod $parent {
    #[path = "$($modulePath.Replace('\', '/'))"]
    mod $module;
}
"@
    [IO.File]::WriteAllText($wrapperPath, $wrapper)
    return $wrapperPath
}

$previousRustupHome = $env:RUSTUP_HOME
$previousToolchain = $env:RUSTUP_TOOLCHAIN
try {
    $env:RUSTUP_HOME = Join-Path $cache 'rustup'
    $env:RUSTUP_TOOLCHAIN = '1.99.1-x86_64-pc-windows-msvc'
    foreach ($name in $files.Keys) {
        & $verus --crate-type lib --no-cheating (Write-Wrapper $name (Join-Path $root $files[$name][0]))
        if ($LASTEXITCODE -ne 0) { throw "Production verification failed: $($files[$name][0])" }
    }

    # Each broken function must fail verification, not merely produce a Rust syntax error.
    $mutants = @(
        @('output', '        x += 1;', '        x += 2;', 'upsampled row that skips output slots'),
        @('output', 'e += 1;', 'e += 2;', 'indexed writer that skips output slots'),
        @('output', 'ox += 4;', 'ox += 5;', 'tiled row that skips output slots'),
        @('output', 'while 4 <= hi - ox', 'while ox < hi', 'tiled row that calls quad outside the interior'),
        @('output', 'p += 1;', 'p += 2;', 'dense tile that skips output columns'),
        @('output', 'while ox < len', 'while ox + 1 < len', 'dense row that skips its last column'),
        @('output', 'while P <= hi - ox', 'while ox < hi', 'dense row that tiles outside the interior'),
        @('output', 'while ox < hi', 'while ox < len', 'dense row that runs unchecked pixels outside the interior'),
        @('unpack', 'if stride < row_bytes {', 'if false {', 'row copy that lets rows overlap'),
        @('unpack', 'if height == 0 {', 'if height <= 1 {', 'row copy that skips a one-row image'),
        @('unpack', 'y += 1;', 'y += 2;', 'row copy that skips rows'),
        @('unpack', 'let at = y * stride;', 'let at = y * row_bytes;', 'row copy that ignores the stride')
    )
    foreach ($m in $mutants) {
        $source = [IO.File]::ReadAllText((Join-Path $root $files[$m[0]][0])).Replace("`r`n", "`n")
        $mutantPath = Join-Path $cache "$($m[0])-mutant.rs"
        [IO.File]::WriteAllText($mutantPath, (Update-SourceOnce $source $m[1] $m[2]))
        $mutantWrapperPath = Write-Wrapper $m[0] $mutantPath
        $ErrorActionPreference = 'Continue'
        & $verus --crate-type lib --no-cheating $mutantWrapperPath *> $log
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -eq 0) { throw "Verus accepted a $($m[3])" }
        if (!(Select-String -Path $log -Pattern 'invariant not satisfied|postcondition not satisfied|precondition not satisfied|assertion failed|callee\.requires' -Quiet)) {
            Get-Content $log
            throw "Mutation failed for an unexpected reason: $($m[3])"
        }
        Write-Output "Verus rejected the $($m[3])"
    }
} finally {
    $env:RUSTUP_HOME = $previousRustupHome
    $env:RUSTUP_TOOLCHAIN = $previousToolchain
}
# The expected mutant failure must not become the calling CI shell's exit code.
exit 0
