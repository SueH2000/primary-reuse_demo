$ErrorActionPreference = "Stop"
$env:PUBLIC_DEMO_MODE="false"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
