$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Installation complete. Run: .\RUN_PUBLIC_DEMO.ps1"
