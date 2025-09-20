@echo off
echo Installing TrafficMonitor...
echo.

REM Create application directory
if not exist "%PROGRAMFILES%\TrafficMonitor" mkdir "%PROGRAMFILES%\TrafficMonitor"

REM Copy executable
copy "dist\TrafficMonitor.exe" "%PROGRAMFILES%\TrafficMonitor\"

REM Create desktop shortcut
echo Creating desktop shortcut...
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut(\"$env:USERPROFILE\Desktop\TrafficMonitor.lnk\"); $Shortcut.TargetPath = \"$env:PROGRAMFILES\TrafficMonitor\TrafficMonitor.exe\"; $Shortcut.Save()"

REM Create start menu shortcut
if not exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\TrafficMonitor" mkdir "%APPDATA%\Microsoft\Windows\Start Menu\Programs\TrafficMonitor"
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut(\"$env:APPDATA\Microsoft\Windows\Start Menu\Programs\TrafficMonitor\TrafficMonitor.lnk\"); $Shortcut.TargetPath = \"$env:PROGRAMFILES\TrafficMonitor\TrafficMonitor.exe\"; $Shortcut.Save()"

echo Installation completed!
echo TrafficMonitor has been installed to: %PROGRAMFILES%\TrafficMonitor
echo Desktop and Start Menu shortcuts have been created.
pause
