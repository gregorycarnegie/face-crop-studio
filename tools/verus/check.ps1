$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$cache = Join-Path $root 'cache/verus'
$verus = Join-Path $cache 'verus-x86-win/verus.exe'
if (!(Test-Path $verus)) { throw 'Run tools/verus/setup.ps1 first' }

function Update-SourceOnce([string] $source, [string] $before, [string] $after) {
    if (($source.Split(@($before), [StringSplitOptions]::None).Length - 1) -ne 1) {
        throw "Expected exactly one source anchor: $before"
    }
    return $source.Replace($before, $after)
}

# Import the entire current production function; insert only its proof annotations.
$source = [IO.File]::ReadAllText((Join-Path $root 'fcs-core/src/cpu/output.rs')).Replace("`r`n", "`n")
$source = [regex]::Replace($source, '(?m)^//!', '//')
$source = (Update-SourceOnce $source '#[cfg(test)]' '// TESTS').Split(@('// TESTS'), [StringSplitOptions]::None)[0]
$source = Update-SourceOnce $source 'pub(super) fn' 'pub fn'
$contract = @'
)
    requires old(output).len() == 2 * input.len(),
    ensures
        final(output).len() == old(output).len(),
        forall|i: int| 0 <= i < final(output).len() ==>
            #[trigger] final(output)[i].as_option() == Some(input[i / 2]),
{
'@
$source = Update-SourceOnce $source ']) {' (']' + $contract)
$invariants = @'
while x < output.len()
        invariant
            output.len() == 2 * input.len(),
            0 <= x <= output.len(),
            forall|i: int| 0 <= i < x ==>
                #[trigger] output[i].as_option() == Some(input[i / 2]),
        decreases output.len() - x,
    {
'@
$source = Update-SourceOnce $source 'while x < output.len() {' $invariants
$proof = "use vstd::prelude::*;`nuse vstd::std_specs::maybe_uninit::MaybeUninitAdditionalSpecFns;`nverus! {`n$source`n}`n"
$proofPath = Join-Path $cache 'output.rs'
[IO.File]::WriteAllText($proofPath, $proof)

$previousRustupHome = $env:RUSTUP_HOME
$previousToolchain = $env:RUSTUP_TOOLCHAIN
try {
    $env:RUSTUP_HOME = Join-Path $cache 'rustup'
    $env:RUSTUP_TOOLCHAIN = '1.99.1-x86_64-pc-windows-msvc'
    & $verus --crate-type lib --no-cheating $proofPath
    if ($LASTEXITCODE -ne 0) { throw 'Production output-row verification failed' }

    # A missed slot must fail verification, not merely produce a Rust syntax error.
    $mutant = Update-SourceOnce $proof 'x += 1;' 'x += 2;'
    $mutantPath = Join-Path $cache 'output-mutant.rs'
    [IO.File]::WriteAllText($mutantPath, $mutant)
    $log = Join-Path $cache 'mutation.log'
    $ErrorActionPreference = 'Continue'
    & $verus --crate-type lib --no-cheating $mutantPath *> $log
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -eq 0) { throw 'Verus accepted a writer that skips output slots' }
    if (!(Select-String -Path $log -Pattern 'invariant not satisfied|postcondition not satisfied' -Quiet)) {
        Get-Content $log
        throw 'Mutation failed for an unexpected reason'
    }
    Write-Output 'Verus rejected the writer that skips output slots'
} finally {
    $env:RUSTUP_HOME = $previousRustupHome
    $env:RUSTUP_TOOLCHAIN = $previousToolchain
}
# The expected mutant failure must not become the calling CI shell's exit code.
exit 0
