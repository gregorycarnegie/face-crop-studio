Unicode True

!ifndef APP_NAME
  !define APP_NAME "Face Crop Studio"
!endif

!ifndef APP_PUBLISHER
  !define APP_PUBLISHER "Face Crop Studio"
!endif

!ifndef APP_VERSION
  !define APP_VERSION "0.0.0"
!endif

!ifndef DIST_DIR
  !error "DIST_DIR define is required"
!endif

!ifndef OUT_FILE
  !error "OUT_FILE define is required"
!endif

Name "${APP_NAME} ${APP_VERSION}"
OutFile "${OUT_FILE}"
InstallDir "$PROGRAMFILES64\Face Crop Studio"
InstallDirRegKey HKLM "Software\Face Crop Studio" "Install_Dir"
RequestExecutionLevel admin

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "WinMessages.nsh"
!include "Sections.nsh"
!include "FileFunc.nsh"
!define MUI_ICON "${DIST_DIR}\app_icon.ico"
!define MUI_UNICON "${DIST_DIR}\app_icon.ico"
!define MUI_LICENSEPAGE_TEXT_TOP "Face Crop Studio is dual-licensed (MIT OR Apache-2.0). Review the terms below."
!define MUI_LICENSEPAGE_CHECKBOX

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${DIST_DIR}\LICENSE-MIT"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

; ── State variables for PATH mutual-exclusion ─────────────────────────────────
Var PathUserWasSelected
Var PathSystemWasSelected

Function .onInit
  StrCpy $PathUserWasSelected 0
  StrCpy $PathSystemWasSelected 0

  ; Detect an existing installation and offer to remove it first so the new
  ; install starts clean rather than silently layering on top.
  ReadRegStr $0 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "UninstallString"
  ${If} $0 != ""
    ReadRegStr $1 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayVersion"
    MessageBox MB_YESNO|MB_ICONQUESTION "Face Crop Studio $1 is already installed.$\n$\nRemove it before installing the new version?" IDNO done
    ExecWait '$0 /S'
    done:
  ${EndIf}
FunctionEnd

; ── Sections ──────────────────────────────────────────────────────────────────

Section "Application Files (required)" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  File /r "${DIST_DIR}\*.*"

  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\Face Crop Studio"
  CreateShortCut "$SMPROGRAMS\Face Crop Studio\Face Crop Studio.lnk" "$INSTDIR\fcs-gui.exe" "" "$INSTDIR\fcs-gui.exe" 0
  CreateShortCut "$SMPROGRAMS\Face Crop Studio\Uninstall Face Crop Studio.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0

  WriteRegStr HKLM "Software\Face Crop Studio" "Install_Dir" "$INSTDIR"

  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "DisplayIcon" "$INSTDIR\fcs-gui.exe"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "NoRepair" 1

  ; Populate EstimatedSize so Add/Remove Programs shows a real disk usage figure.
  ; GetSize returns KB; measured after all files are written so it is accurate.
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "EstimatedSize" $0
SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortCut "$DESKTOP\Face Crop Studio.lnk" "$INSTDIR\fcs-gui.exe" "" "$INSTDIR\fcs-gui.exe" 0
SectionEnd

Section /o "Add install directory to PATH (Current user)" SecPathUser
  Call AddInstallDirToUserPath
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathUser" 1
  Call BroadcastEnvironmentChange
SectionEnd

Section /o "Add install directory to PATH (All users)" SecPathSystem
  Call AddInstallDirToSystemPath
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathSystem" 1
  Call BroadcastEnvironmentChange
SectionEnd

Section "Uninstall"
  InitPluginsDir

  ReadRegDWORD $0 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathUser"
  ${If} $0 == 1
    Call un.RemoveInstallDirFromUserPath
  ${EndIf}

  ReadRegDWORD $1 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio" "AddPathSystem"
  ${If} $1 == 1
    Call un.RemoveInstallDirFromSystemPath
  ${EndIf}

  Call un.BroadcastEnvironmentChange

  Delete "$DESKTOP\Face Crop Studio.lnk"
  Delete "$SMPROGRAMS\Face Crop Studio\Face Crop Studio.lnk"
  Delete "$SMPROGRAMS\Face Crop Studio\Uninstall Face Crop Studio.lnk"
  RMDir "$SMPROGRAMS\Face Crop Studio"

  ; Delete installed files by pattern so this section stays correct as binaries
  ; and assets are added or renamed without needing manual updates here.
  ; Wildcards cover all present and future executables / icons / docs.
  Delete "$INSTDIR\*.exe"     ; fcs-gui.exe, fcs-cli.exe, Uninstall.exe
  Delete "$INSTDIR\*.dll"     ; onnxruntime.dll
  Delete "$INSTDIR\*.ico"
  Delete "$INSTDIR\*.md"
  Delete "$INSTDIR\LICENSE-*"
  ; Remove only the bundled models; RMDir (no /r) leaves the directory intact
  ; if the user placed additional models there.
  Delete "$INSTDIR\models\face_detection_yunet_2023mar_640.onnx"
  Delete "$INSTDIR\models\eye_refiner.onnx"
  RMDir "$INSTDIR\models"
  ; Remove bundled sample images and docs; RMDir (no /r) preserves any extra
  ; files the user dropped into the samples folder.
  Delete "$INSTDIR\samples\*.jpg"
  Delete "$INSTDIR\samples\*.png"
  Delete "$INSTDIR\samples\*.webp"
  Delete "$INSTDIR\samples\*.md"
  RMDir "$INSTDIR\samples"
  ; Remove the install directory only if it is now empty.
  RMDir "$INSTDIR"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\FaceCropStudio"
  DeleteRegKey HKLM "Software\Face Crop Studio"
SectionEnd

; ── Section descriptions (shown on the Components page) ───────────────────────
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain}       "Core application files — fcs-gui.exe, fcs-cli.exe, bundled model, and required assets. Cannot be deselected."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop}    "Place a shortcut to Face Crop Studio on the desktop."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecPathUser}   "Append the install directory to the current user's PATH so fcs-cli is available in new terminals."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecPathSystem} "Append the install directory to the system PATH (all users). Requires administrator rights — mutually exclusive with the per-user option."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ── PATH helpers ──────────────────────────────────────────────────────────────
;
; Each function writes a small PowerShell script to $PLUGINSDIR (a per-session
; temp dir that NSIS cleans up on exit), executes it, and returns.  Keeping the
; logic in readable multi-line PS1 files is far easier to audit than embedding
; escaped one-liners inside nsExec strings.

Function AddInstallDirToUserPath
  ; Append $INSTDIR to the current user PATH, skipping if already present.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "User"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" } } else { @() }$\r$\n'
  FileWrite $R8 'if ($$parts -notcontains $$dir) {$\r$\n'
  FileWrite $R8 '    $$parts += $$dir$\r$\n'
  FileWrite $R8 '    [Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileWrite $R8 '}$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

Function AddInstallDirToSystemPath
  ; Append $INSTDIR to the system (all-users) PATH, skipping if already present.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "Machine"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" } } else { @() }$\r$\n'
  FileWrite $R8 'if ($$parts -notcontains $$dir) {$\r$\n'
  FileWrite $R8 '    $$parts += $$dir$\r$\n'
  FileWrite $R8 '    [Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileWrite $R8 '}$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

Function un.RemoveInstallDirFromUserPath
  ; Remove $INSTDIR from the current user PATH, leaving other entries intact.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "User"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" -and $$_ -ne $$dir } } else { @() }$\r$\n'
  FileWrite $R8 '[Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

Function un.RemoveInstallDirFromSystemPath
  ; Remove $INSTDIR from the system (all-users) PATH, leaving other entries intact.
  FileOpen $R8 "$PLUGINSDIR\fcs_path.ps1" w
  FileWrite $R8 '$$scope = "Machine"$\r$\n'
  FileWrite $R8 '$$dir   = "$INSTDIR"$\r$\n'
  FileWrite $R8 '$$cur   = [Environment]::GetEnvironmentVariable("Path", $$scope)$\r$\n'
  FileWrite $R8 '$$parts = if ($$cur) { $$cur.Split(";") | Where-Object { $$_.Trim() -ne "" -and $$_ -ne $$dir } } else { @() }$\r$\n'
  FileWrite $R8 '[Environment]::SetEnvironmentVariable("Path", ($$parts -join ";"), $$scope)$\r$\n'
  FileClose $R8
  nsExec::ExecToLog 'powershell -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\fcs_path.ps1"'
FunctionEnd

; ── Utility functions ─────────────────────────────────────────────────────────

Function BroadcastEnvironmentChange
  ; Notify running processes that the environment has changed (makes PATH
  ; visible in new terminals without requiring a logoff/reboot).
  System::Call 'USER32::SendMessageTimeout(p ${HWND_BROADCAST}, i ${WM_SETTINGCHANGE}, p 0, t "Environment", i 0, i 5000, *p .r0)'
FunctionEnd

Function un.BroadcastEnvironmentChange
  System::Call 'USER32::SendMessageTimeout(p ${HWND_BROADCAST}, i ${WM_SETTINGCHANGE}, p 0, t "Environment", i 0, i 5000, *p .r0)'
FunctionEnd

; ── Mutual exclusion: prevent user and system PATH from both being selected ───
;
; "Last clicked wins": $PathUserWasSelected / $PathSystemWasSelected track the
; previous state so we can tell which section was just activated and deselect
; the other one rather than always deselecting the same side.

Function .onSelChange
  SectionGetFlags ${SecPathUser}   $0
  SectionGetFlags ${SecPathSystem} $1
  IntOp $2 $0 & ${SF_SELECTED}   ; 0 or 1 — user currently selected?
  IntOp $3 $1 & ${SF_SELECTED}   ; 0 or 1 — system currently selected?

  ${If} $2 <> 0
  ${AndIf} $3 <> 0
    ; Both on — keep the one that just changed, deselect the other.
    ${If} $PathUserWasSelected = 0
      ; User PATH was off before → it was just turned on → clear System PATH
      IntOp $1 $1 & ~${SF_SELECTED}
      SectionSetFlags ${SecPathSystem} $1
      StrCpy $PathUserWasSelected 1
      StrCpy $PathSystemWasSelected 0
    ${Else}
      ; User PATH was already on → System PATH was just turned on → clear User PATH
      IntOp $0 $0 & ~${SF_SELECTED}
      SectionSetFlags ${SecPathUser} $0
      StrCpy $PathUserWasSelected 0
      StrCpy $PathSystemWasSelected 1
    ${EndIf}
  ${Else}
    ; No conflict — just keep prev-state in sync for the next call.
    StrCpy $PathUserWasSelected $2
    StrCpy $PathSystemWasSelected $3
  ${EndIf}
FunctionEnd
