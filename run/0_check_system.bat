@echo off
cd /d "%~dp0.."
set LOG=run\system_check.txt
echo ==== AgriSmart system check %date% %time% > %LOG%
echo --- python on PATH >> %LOG%
where python >> %LOG% 2>&1
python --version >> %LOG% 2>&1
echo --- anaconda python >> %LOG%
if exist "%USERPROFILE%\anaconda3\python.exe" ("%USERPROFILE%\anaconda3\python.exe" --version >> %LOG% 2>&1) else (echo none >> %LOG%)
echo --- py launcher >> %LOG%
py -0 >> %LOG% 2>&1
echo --- mysql client >> %LOG%
where mysql >> %LOG% 2>&1
echo --- mysql services >> %LOG%
sc query state= all | findstr /I "mysql" >> %LOG% 2>&1
echo --- mysql program files >> %LOG%
dir /b "C:\Program Files\MySQL" >> %LOG% 2>&1
echo --- winget >> %LOG%
winget --version >> %LOG% 2>&1
echo --- vscode >> %LOG%
where code >> %LOG% 2>&1
echo ==== done >> %LOG%
