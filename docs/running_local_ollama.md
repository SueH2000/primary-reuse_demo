# Running with local Ollama

The public demo does not require Ollama. Ollama is only needed for optional LLM review modes.

## 1. Install Ollama

Install Ollama from the official Ollama website, then verify in PowerShell:

```powershell
ollama --version
```

## 2. Pull a model

```powershell
ollama pull llama3
```

## 3. Start the backend

```powershell
$env:CLASSIFIER_OLLAMA_MODEL="llama3"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## 4. Use LLM modes in the UI

In the execution flow panel, select one of:

- Sentence judge routed
- Sentence judge force
- Final classify routed
- Final classify force

Baseline modes do not call Ollama.

## Security note

Do not expose raw Ollama directly to the public internet. Keep it local unless you implement authentication and a backend proxy.
