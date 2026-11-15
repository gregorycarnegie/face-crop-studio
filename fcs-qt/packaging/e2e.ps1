# End-to-end run of a built fcs-qt.exe, driven through UI Automation as a user would: open it on
# a folder of photos, wait until every face is found, export them all through the folder dialog,
# check one crop arrived per selected face, then close the window and check Qt shut down cleanly.
#   ./fcs-qt/packaging/e2e.ps1 fcs-qt/dist/face-crop-studio-qt-windows-x86_64/fcs-qt.exe
# Needs an interactive desktop (the CI runners have one). Keys are only sent while fcs-qt owns
# the foreground window.
param([Parameter(Mandatory)][string]$Exe, [int]$Photos = 6)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms, System.Drawing
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class E2E {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint f);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint x, uint y, uint d, UIntPtr e);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  public struct RECT { public int L, T, R, B; }
}
"@
$exe = (Resolve-Path $Exe).Path
# Under RUNNER_TEMP on CI, where the failure screenshots are collected from.
$temp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$work = New-Item -ItemType Directory -Force (Join-Path $temp "fcs-qt-e2e-$PID")
$inbox = New-Item -ItemType Directory -Force "$work\photos"
$outbox = New-Item -ItemType Directory -Force "$work\crops"
Get-ChildItem "$PSScriptRoot\..\..\fixtures\images\*.jpg" | Sort-Object Name | Select-Object -First $Photos |
    Copy-Item -Destination $inbox
$auto = [Windows.Automation.AutomationElement]
$tree = [Windows.Automation.TreeScope]

function Wait-For([string]$what, [scriptblock]$test, [int]$seconds = 120) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $value = & $test
        if ($value) { return $value }
        Start-Sleep -Milliseconds 250
    }
    throw "e2e: timed out after $seconds s waiting for $what"
}
function Find-Named($root, [string]$pattern) {
    $all = $root.FindAll($tree::Descendants, [Windows.Automation.Condition]::TrueCondition)
    foreach ($e in $all) { if ($e.Current.Name -match $pattern) { return $e } }
}
# On failure: every window of the app, as $work\failure-<n>.png (CI uploads them).
function Save-Screens($process) {
    $n = 0
    foreach ($w in $auto::RootElement.FindAll($tree::Children, (New-Object Windows.Automation.PropertyCondition($auto::ProcessIdProperty, $process.Id)))) {
        foreach ($e in @($w) + @($w.FindAll($tree::Children, (New-Object Windows.Automation.PropertyCondition($auto::ControlTypeProperty, [Windows.Automation.ControlType]::Window))))) {
            $h = [IntPtr]$e.Current.NativeWindowHandle; $r = New-Object E2E+RECT
            if ($h -eq 0 -or -not [E2E]::GetWindowRect($h, [ref]$r) -or $r.R -le $r.L) { continue }
            $bmp = New-Object Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
            $g = [Drawing.Graphics]::FromImage($bmp); $dc = $g.GetHdc()
            [E2E]::PrintWindow($h, $dc, 2) | Out-Null; $g.ReleaseHdc($dc)
            $bmp.Save("$work\failure-$n.png"); $n++
        }
    }
    "e2e: screenshots in $work"
}
function Send-Keys($process, [string]$keys) {
    $fg = [E2E]::GetForegroundWindow(); $owner = 0
    [E2E]::GetWindowThreadProcessId($fg, [ref]$owner) | Out-Null
    if ($owner -ne $process.Id) { throw "e2e: fcs-qt lost the foreground; not sending '$keys'" }
    [Windows.Forms.SendKeys]::SendWait($keys)
}

$log = "$work\stderr.log"
$p = Start-Process $exe -ArgumentList "`"$inbox`"" -WorkingDirectory $work -RedirectStandardError $log -PassThru
try {
    # Qt shows a helper window first, so wait for the titled one that holds the photo list.
    $mine = New-Object Windows.Automation.PropertyCondition($auto::ProcessIdProperty, $p.Id)
    $window = Wait-For 'the main window' {
        $auto::RootElement.FindAll($tree::Children, $mine) |
            Where-Object { $_.Current.Name -eq 'Face Crop Studio' -and (Find-Named $_ '^Photos$') } |
            Select-Object -First 1
    } 60
    $hwnd = [IntPtr]$window.Current.NativeWindowHandle
    $list = Find-Named $window '^Photos$'
    Wait-For "all $Photos photos to be ready" { Find-Named $window '^All photos ready$' } | Out-Null
    $rows = $list.FindAll($tree::Children, [Windows.Automation.Condition]::TrueCondition).Count
    if ($rows -lt $Photos) { throw "e2e: $rows items in the photo list, expected at least $Photos" }
    $button = Wait-For 'the export button' { Find-Named $window '^Export \d+ faces?$' }
    $faces = [int]($button.Current.Name -replace '\D', '')
    if ($faces -lt 1) { throw 'e2e: no faces found' }
    "found $faces faces in $Photos photos"

    [E2E]::SetForegroundWindow($hwnd) | Out-Null
    $button.GetCurrentPattern([Windows.Automation.InvokePattern]::Pattern).Invoke()
    # UI Automation lists an owned dialog under its owner window, not the desktop.
    function Find-Dialog {
        @($auto::RootElement.FindAll($tree::Children, $mine)) + @($window.FindAll($tree::Children, [Windows.Automation.Condition]::TrueCondition)) |
            Where-Object { $_.Current.Name -eq 'Export faces to' } | Select-Object -First 1
    }
    $dialog = Wait-For 'the folder dialog' { Find-Dialog } 30
    # Through the address bar (Alt+D), which every layout of the folder picker has, then
    # "Select Folder". SendKeys treats + ^ % ~ ( ) { } [ ] as commands, and runner temp paths
    # contain ~, so each is sent in braces.
    $dialogHwnd = [IntPtr]$dialog.Current.NativeWindowHandle
    [E2E]::SetForegroundWindow($dialogHwnd) | Out-Null
    Start-Sleep -Milliseconds 500
    $path = [regex]::Replace($outbox.FullName, '[+^%~(){}\[\]]', { '{' + $args[0].Value + '}' })
    Send-Keys $p '%d'
    Start-Sleep -Milliseconds 500
    Send-Keys $p "$path{ENTER}"
    Start-Sleep 2
    $select = Wait-For 'the Select Folder button' { Find-Named $dialog '^Select Folder$' } 10
    try { $select.GetCurrentPattern([Windows.Automation.InvokePattern]::Pattern).Invoke() } catch { }
    Start-Sleep 1
    if (Find-Dialog) {
        # Its Invoke can be a no-op; a click at its centre is what a user does anyway.
        $box = $select.Current.BoundingRectangle
        [E2E]::SetForegroundWindow($dialogHwnd) | Out-Null
        Start-Sleep -Milliseconds 300
        Send-Keys $p ''   # only the foreground check
        [E2E]::SetCursorPos([int]($box.X + $box.Width / 2), [int]($box.Y + $box.Height / 2)) | Out-Null
        [E2E]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero); [E2E]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
    }
    Wait-For 'the dialog to close' { -not (Find-Dialog) } 30 | Out-Null

    Wait-For "$faces crops" { (Get-ChildItem $outbox -File -Recurse | Where-Object Extension -match '\.(jpe?g|png|webp|tiff?|bmp|avif)$').Count -ge $faces } | Out-Null
    Wait-For 'the export to finish' { Find-Named $window '^Export \d+ faces?$' } 60 | Out-Null
    $crops = (Get-ChildItem $outbox -File -Recurse | Where-Object Extension -match '\.(jpe?g|png|webp|tiff?|bmp|avif)$').Count
    if ($crops -ne $faces) { throw "e2e: $crops crops for $faces faces" }
    "exported $crops crops"

    [E2E]::SendMessage($hwnd, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null   # WM_CLOSE
    if (-not $p.WaitForExit(30000)) { throw 'e2e: the app did not exit within 30 s of closing' }
} catch {
    if (-not $p.HasExited) { Save-Screens $p }
    throw
} finally {
    if (-not $p.HasExited) { Stop-Process $p -Force }
}
$text = Get-Content $log -Raw
if ($text -match 'QDxgiVSyncService|QThreadStorage|qrc:/|is not installed|Cannot load library|panicked') {
    $text
    throw 'e2e: errors or shutdown warnings on stderr (above)'
}
'e2e: ok'
