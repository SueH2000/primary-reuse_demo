# Set OLLAMA_API_KEY in this PowerShell session before running this script:
# $env:OLLAMA_API_KEY="your_new_key"
$env:PUBLIC_DEMO_MODE="true"
$env:USE_SUPABASE_FEEDBACK="false"
$env:USE_OLLAMA="true"
$env:CLASSIFIER_OLLAMA_URL="https://ollama.com/api/generate"
$env:CLASSIFIER_OLLAMA_MODEL="gpt-oss:20b-cloud"
$env:CLASSIFIER_LLM_OVERRIDE_LOCK_THRESHOLD="0.85"
if (-not $env:OLLAMA_API_KEY) { Write-Host "OLLAMA_API_KEY is not set."; exit 1 }
uvicorn classification_api:app --host 127.0.0.1 --port 8000
