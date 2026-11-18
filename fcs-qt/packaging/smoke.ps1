# Windows twin of smoke.sh: run the packaged fcs-qt.exe on the samples for at least 20 s, from an
# empty folder and with Qt off PATH, and check it stayed up, loaded the detector and its QML.
#   ./fcs-qt/packaging/smoke.ps1 fcs-qt/dist/face-crop-studio-qt-windows-x86_64/fcs-qt.exe
param([Parameter(Mandatory)][string]$Exe)
$ErrorActionPreference = 'Stop'
$exe = (Resolve-Path $Exe).Path
$samples = (Resolve-Path "$PSScriptRoot\..\..\samples").Path
$work = New-Item -ItemType Directory -Force (Join-Path ([IO.Path]::GetTempPath()) "fcs-qt-smoke-$PID")
$log = "$work\stderr.log"
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"   # only what a user's machine has
$p = Start-Process $exe -ArgumentList "`"$samples`"" -WorkingDirectory $work -RedirectStandardError $log -PassThru
# 20 s up at least, then up to 90 s for the detector: on CI's WARP adapter it has missed 20 s.
$start = Get-Date
do {
    Start-Sleep 2
    $text = try {   # ReadWrite share: the app still has the log open for writing
        $s = [IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
        try { [IO.StreamReader]::new($s).ReadToEnd() } finally { $s.Dispose() }
    } catch { '' }
    $elapsed = ((Get-Date) - $start).TotalSeconds
} until ($p.HasExited -or ($elapsed -ge 20 -and $text -match 'SCRFD on') -or $elapsed -ge 90)
$alive = -not $p.HasExited
if ($alive) { Stop-Process $p -Force; $p.WaitForExit() }
$text = Get-Content $log -Raw
$text
"smoke: stopped after $([int]$elapsed) s"
if (-not $alive) { throw "smoke: the app exited early (code $($p.ExitCode))" }
if ($text -notmatch 'SCRFD on') { throw 'smoke: the detector did not load from the package' }
if ($text -match 'qrc:/|is not installed|Cannot load library|FXC D3DCompile error') {
    throw 'smoke: QML, plugin or shader errors (above)'
}
'smoke: ok'
