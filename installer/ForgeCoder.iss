; ForgeCoder — Windows installer (Inno Setup 6)
;
; Install layout:
;   C:\Program Files\ForgeCoder\
;       forge-server.exe      (wrapper: python -m forge_server.main)
;       llama-server.exe
;       forge-indexer.exe     (wrapper: forge-index)
;       models\forgecoder-1.5b-q4_k_m.gguf
;   %LOCALAPPDATA%\ForgeCoder\
;       config.json
;       forge.db
;       logs\
;
; Build: iscc installer\ForgeCoder.iss

#ifndef InnoSetupVersion
  #define InnoSetupVersion "6.2.2"
#endif

#define AppName "ForgeCoder"
#define AppVersion "0.1.0"
#define AppPublisher "ForgeCoder Contributors"
#define AppId "{{3F7B9C2E-8D41-4E0A-9A5E-FORGE001}"
#define ModelFile "..\models\gguf\forgecoder-1.5b-q4_k_m.gguf"

[Setup]
AppId={{3F7B9C2E-8D41-4E0A-9A5E-B1FC00F00D01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\ForgeCoder
DefaultGroupName=ForgeCoder
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=ForgeCoder-Setup-{#AppVersion}
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\assets\forge.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "vscode"; Description: "Install the VS Code extension"; GroupDescription: "Integrations:"; Flags: checkedonce

[Files]
Source: "..\runtime\llama\llama-server.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\runtime\llama\llama-quantize.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ModelFile}"; DestDir: "{app}\models"; DestName: "forgecoder-1.5b-q4_k_m.gguf"; Check: ModelExists
Source: "..\apps\vscode\forgecoder-0.1.0.vsix"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: VsixExists
Source: "assets\forge.ico"; DestDir: "{app}\assets"; Flags: ignoreversion

; Python resources (the installed tree mirrors the repo's package layout).
Source: "..\pyproject.toml"; DestDir: "{app}\resources"
Source: "..\core\*.py"; DestDir: "{app}\resources\core"; Flags: recursesubdirs
Source: "..\apps\server\forge_server\*.py"; DestDir: "{app}\resources\forge_server"
Source: "..\runtime\prompts\*.txt"; DestDir: "{app}\resources\runtime\prompts"
Source: "..\runtime\grammars\*.gbnf"; DestDir: "{app}\resources\runtime\grammars"

[Icons]
Name: "{autoprograms}\ForgeCoder"; Filename: "{app}\launch-forge.cmd"
Name: "{autoprograms}\ForgeCoder (run minimized)"; Filename: "{app}\launch-forge.cmd"; Parameters: "/minimized"; Flags: runminimized

[Registry]
Root: HKCU; Subkey: "Software\ForgeCoder"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}"; Flags: uninsdeletekey

[Run]
Filename: "{app}\launch-forge.cmd"; WorkingDir: "{app}"; Flags: nowait runhidden postinstall skipifsilent
Filename: "{cmd}"; Parameters: "/c code --install-extension {tmp}\forgecoder-0.1.0.vsix"; Flags: runhidden; Tasks: vscode

[UninstallRun]
Filename: "{cmd}"; Parameters: "/c taskkill /f /im forge-server.exe >nul 2>&1"; Flags: runhidden

[Code]
var
  ModelMissing: Boolean;
  VsixMissing: Boolean;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not IsWin64 then begin
    MsgBox('ForgeCoder requires 64-bit Windows 10 or 11.', mbError, MB_OK);
    Result := False;
    Exit;
  end;
  if not IsDiskSpaceEnough(6 * 1024 * 1024 * 1024) then
    MsgBox('The model needs about 1.2 GB; your disk is low on space.', mbWarning, MB_OK);
  ModelMissing := not FileExists(ExpandConstant('{#ModelFile}'));
  if ModelMissing then
    MsgBox('No model found at models\gguf\forgecoder-1.5b-q4_k_m.gguf.' + #13#10 +
           'Setup will continue, but inference will not work until a model is added.', mbWarning, MB_OK);
  VsixMissing := not FileExists(ExpandConstant('..\apps\vscode\forgecoder-0.1.0.vsix'));
  if VsixMissing then
    MsgBox('The VS Code extension VSIX was not found; the integration step will be skipped.', mbWarning, MB_OK);
  Result := True;
end;

function ModelExists(): Boolean;
begin
  Result := not ModelMissing;
end;

function VsixExists(): Boolean;
begin
  Result := not VsixMissing;
end;