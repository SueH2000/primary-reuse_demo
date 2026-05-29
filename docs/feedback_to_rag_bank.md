# Feedback-to-RAG-Bank Workflow

The public demo collects feedback so that useful corrections can later improve the RAG bank. The flow is deliberately gated:

```text
public feedback
  -> Supabase rag_feedback table, review_status = pending
  -> curator review
  -> review_status = approved and approved_for_rag = true
  -> export approved feedback to rag_feedback_gold_standard.csv
  -> refresh_rag_bank.py
  -> rag_bank_refreshed.csv
```

## Why there is a review gate

Do not write public feedback directly into the gold standard. Public feedback may be wrong, duplicated, contradictory, or unsupported by evidence. The RAG bank should contain only curator-approved corrections.

## 1. Curate in Supabase

In the Supabase table editor, inspect each feedback row. Approve only rows that have a defensible correction and evidence.

For approved rows, set:

```text
review_status = approved
approved_for_rag = true
reviewed_at = current timestamp
curator_note = your note
```

Reject weak rows:

```text
review_status = rejected
approved_for_rag = false
```

## 2. Export approved feedback

Set your environment variables locally:

```powershell
$env:USE_SUPABASE_FEEDBACK="true"
$env:SUPABASE_URL="https://YOUR_PROJECT_REF.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
$env:SUPABASE_FEEDBACK_TABLE="rag_feedback"
```

Then run:

```powershell
python export_approved_feedback.py --output-csv rag_feedback_gold_standard.csv
```

## 3. Refresh the RAG bank

Use your private base labeled CSV locally:

```powershell
python refresh_rag_bank.py `
  --base-csv manual_ground_truth_with_GSE_links_REFRESHED.csv `
  --feedback-csv rag_feedback_gold_standard.csv `
  --output-csv rag_bank_refreshed.csv `
  --report-json rag_bank_refresh_report.json
```

Only after this step should the refreshed bank be used in a new deployment or benchmark.

## 4. Public wording

Use this wording in presentations:

> The public demo collects reviewer feedback into a pending feedback database. Feedback is not automatically used for model updates. After manual curation, approved corrections can be exported and merged into a refreshed RAG bank.
