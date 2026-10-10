# AgriSmart - one-time setup on Windows 10/11 (PowerShell)
# Run from the project folder:   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== 1. Checking for MySQL Server" -ForegroundColor Green
$svc = Get-Service -Name "MySQL*" -ErrorAction SilentlyContinue
$cli = Get-Command mysql -ErrorAction SilentlyContinue
if (-not $svc -and -not $cli) {
    Write-Host "MySQL not found. Installing MySQL Server with winget..." -ForegroundColor Yellow
    winget install --id Oracle.MySQL -e --accept-package-agreements --accept-source-agreements
    Write-Host ""
    Write-Host "MySQL was installed. If the MySQL Configurator / Installer opens, choose:" -ForegroundColor Yellow
    Write-Host "  * Development Computer, port 3306"
    Write-Host "  * set a ROOT password and remember it"
    Write-Host "  * Configure MySQL Server as a Windows Service (MySQL80), start at system startup"
    Write-Host "If winget is not available, download 'MySQL Installer for Windows' from https://dev.mysql.com/downloads/installer/"
    Read-Host "Press Enter once MySQL is configured and running"
} else {
    Write-Host "MySQL found: $($svc.Name) $($cli.Source)"
    if ($svc -and $svc.Status -ne "Running") { Start-Service $svc.Name; Write-Host "Started $($svc.Name)" }
}

Write-Host "== 2. Python virtual environment + packages" -ForegroundColor Green
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "== 2b. Train the models if they are not present (about 3 minutes)" -ForegroundColor Green
if (-not (Test-Path "ml\models\crop_model_with_pH.joblib")) {
    Push-Location ml
    & ..\.venv\Scripts\python.exe -W ignore train_models.py
    Pop-Location
} else { Write-Host "Models already present." }

Write-Host "== 3. .env configuration" -ForegroundColor Green
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    $pw = Read-Host "Enter your MySQL root password (stored only in .env)"
    (Get-Content ".env") -replace "^DB_ADMIN_PASSWORD=.*", "DB_ADMIN_PASSWORD=$pw" | Set-Content ".env"
}

Write-Host "== 4. Create database, tables, views, triggers, procedures, demo data" -ForegroundColor Green
& .\.venv\Scripts\python.exe scripts\setup_database.py

Write-Host ""
Write-Host "Done. Start the website with:   .\.venv\Scripts\python.exe -m webapp.app" -ForegroundColor Green
Write-Host "Then open http://127.0.0.1:5000"
