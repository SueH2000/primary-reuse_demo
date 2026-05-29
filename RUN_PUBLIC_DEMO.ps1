$ErrorActionPreference = "Stop"
$env:PUBLIC_DEMO_MODE="true"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
