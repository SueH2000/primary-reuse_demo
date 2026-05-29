$ErrorActionPreference = "Stop"
$env:PUBLIC_DEMO_MODE="true"
$env:USE_SUPABASE_FEEDBACK="false"
$env:CLASSIFIER_OLLAMA_MODEL=${env:CLASSIFIER_OLLAMA_MODEL}
if ([string]::IsNullOrWhiteSpace($env:CLASSIFIER_OLLAMA_MODEL)) {
  $env:CLASSIFIER_OLLAMA_MODEL="llama3:latest"
}
$env:CLASSIFIER_OLLAMA_URL="http://localhost:11434/api/generate"
$env:CLASSIFIER_LLM_OVERRIDE_LOCK_THRESHOLD="0.85"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
