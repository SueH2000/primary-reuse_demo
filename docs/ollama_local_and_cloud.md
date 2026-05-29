# Local Ollama and Ollama Cloud

This project supports two Ollama backends:

1. Local Ollama on your machine.
2. Ollama Cloud from a hosted deployment such as Render.

## Local Ollama

Use this for private local testing.

```powershell
$env:USE_OLLAMA="true"
$env:CLASSIFIER_OLLAMA_URL="http://localhost:11434/api/generate"
$env:CLASSIFIER_OLLAMA_MODEL="llama3:latest"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

## Ollama Cloud

Use this for Render or another public backend. Do not commit the API key.

```powershell
$env:USE_OLLAMA="true"
$env:CLASSIFIER_OLLAMA_URL="https://ollama.com/api/generate"
$env:CLASSIFIER_OLLAMA_MODEL="gpt-oss:20b-cloud"
$env:OLLAMA_API_KEY="your_new_key"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

For Render, set the same values in Environment Variables:

```text
USE_OLLAMA=true
CLASSIFIER_OLLAMA_URL=https://ollama.com/api/generate
CLASSIFIER_OLLAMA_MODEL=gpt-oss:20b-cloud
OLLAMA_API_KEY=<secret>
```

The application automatically adds `Authorization: Bearer <OLLAMA_API_KEY>` when `OLLAMA_API_KEY` is present. Without that key, the same code works with local Ollama.

## Recommended demo setting

Public Render demo:

```text
PUBLIC_DEMO_MODE=true
USE_SUPABASE_FEEDBACK=true
USE_OLLAMA=true
CLASSIFIER_OLLAMA_URL=https://ollama.com/api/generate
CLASSIFIER_OLLAMA_MODEL=gpt-oss:20b-cloud
OLLAMA_API_KEY=<secret>
CLASSIFIER_LLM_OVERRIDE_LOCK_THRESHOLD=0.85
```

Local private demo:

```text
PUBLIC_DEMO_MODE=true
USE_SUPABASE_FEEDBACK=false
USE_OLLAMA=true
CLASSIFIER_OLLAMA_URL=http://localhost:11434/api/generate
CLASSIFIER_OLLAMA_MODEL=llama3:latest
CLASSIFIER_LLM_OVERRIDE_LOCK_THRESHOLD=0.85
```
