# Primary vs Reuse Provenance Review Console

Evidence-backed review console for classifying biomedical papers as `Primary` or `Reuse` based on data provenance.

The app keeps the original FastAPI + static web UI architecture:

```text
FastAPI backend
+ webui/index.html
+ webui/app.js
+ webui/app.css
+ production_classifier.py
+ evidence_modeling.py
```

It supports:

- pasted article text classification;
- local identifier / PMCID lookup;
- batch upload;
- GEO/GSE accession extraction;
- baseline and LLM-mode comparison;
- reviewer feedback collection;
- optional permanent public deployment with Supabase feedback staging;
- curated feedback export into a refreshed RAG bank.

## Important design rule

Public feedback does **not** directly update the gold standard or RAG bank.

The intended workflow is:

```text
public feedback
  -> pending feedback database
  -> curator review
  -> approved feedback export
  -> refresh_rag_bank.py
  -> refreshed RAG bank
```

This prevents low-quality public feedback from contaminating the retrieval bank.

## Run locally on PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:PUBLIC_DEMO_MODE="true"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## Local full version with private data

Put the private files in the repository root locally:

```text
manual_ground_truth_with_GSE_links_REFRESHED.csv
pmc_gse_articles.jsonl
Mohammad_doi.csv              # optional
```

Then run:

```powershell
$env:PUBLIC_DEMO_MODE="false"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

Do not push private files to GitHub.

## Permanent public demo with feedback collection

Recommended free/low-cost setup:

```text
GitHub repo
  -> Render Web Service runs FastAPI
  -> Supabase stores pending feedback
```

See:

```text
docs/deploy_render_supabase.md
```

The public deployment should use:

```text
PUBLIC_DEMO_MODE=true
USE_OLLAMA=false
USE_SUPABASE_FEEDBACK=true
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_FEEDBACK_TABLE=rag_feedback
```

Set the real Supabase values in Render Environment Variables. Do not commit them.

## Export approved feedback and refresh the RAG bank

After curating feedback in Supabase, export only approved rows:

```powershell
$env:USE_SUPABASE_FEEDBACK="true"
$env:SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
$env:SUPABASE_FEEDBACK_TABLE="rag_feedback"

python export_approved_feedback.py --output-csv rag_feedback_gold_standard.csv
```

Then refresh the RAG bank locally:

```powershell
python refresh_rag_bank.py `
  --base-csv manual_ground_truth_with_GSE_links_REFRESHED.csv `
  --feedback-csv rag_feedback_gold_standard.csv `
  --output-csv rag_bank_refreshed.csv `
  --report-json rag_bank_refresh_report.json
```

See:

```text
docs/feedback_to_rag_bank.md
```

## Public demo data

The public package includes synthetic demo data:

```text
data/demo_labels.csv
data/demo_articles.jsonl
```

These are only for deployment safety. They are not the real training/evaluation dataset.

## Local Ollama

Local Ollama is optional. The public demo should not depend on Ollama.

For local testing:

```powershell
ollama pull llama3
$env:PUBLIC_DEMO_MODE="false"
$env:USE_OLLAMA="true"
$env:CLASSIFIER_OLLAMA_MODEL="llama3"
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

## Do not publish these files

The `.gitignore` blocks these private/generated files:

```text
manual_ground_truth_with_GSE_links_REFRESHED.csv
pmc_gse_articles.jsonl
Mohammad_doi.csv
rag_feedback_gold_standard.csv
rag_bank_refreshed.csv
runtime_state/
cache/
.venv/
.env
```

## Roadmap

Near-term:

1. Keep the public provenance review console online.
2. Collect pending feedback through Supabase.
3. Curate approved feedback and refresh the RAG bank.
4. Add GEO/SRA metadata connectors.
5. Add biological experiment extraction.
6. Add computational method extraction.
7. Build protocol-level reproducibility audit reports.

## LLM Guardrail Policy

The public demo uses `linear_model_plus_rag` as the stable decision path. LLM modes are reviewer aids. If the baseline prediction is high-confidence auto-accept, the LLM may still be called for diagnostic comparison, but it cannot overwrite the final label. This prevents direct LLM classification from changing strong evidence-backed decisions. See `docs/llm_guardrails.md`.

## Ollama backend modes

The optional LLM reviewer supports both local Ollama and Ollama Cloud.

- Local Ollama: `CLASSIFIER_OLLAMA_URL=http://localhost:11434/api/generate`, no API key.
- Ollama Cloud: `CLASSIFIER_OLLAMA_URL=https://ollama.com/api/generate`, `CLASSIFIER_OLLAMA_MODEL=gpt-oss:20b-cloud`, and `OLLAMA_API_KEY` set in the backend environment.

See `docs/ollama_local_and_cloud.md`.
