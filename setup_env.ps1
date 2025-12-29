# Topological Circuit Complexity - Environment Setup Script
# Run this script to create and configure the virtual environment

$ErrorActionPreference = "Stop"

Write-Host "=== Topological Circuit Complexity - Environment Setup ===" -ForegroundColor Cyan

# Create virtual environment
Write-Host "`n[1/3] Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# Activate virtual environment
Write-Host "[2/3] Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "[3/3] Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`nTo verify the installation, run:" -ForegroundColor Cyan
Write-Host "  python verify_env.py" -ForegroundColor White
