# Public demo mode

Public demo mode makes the repository safe to run without private labeled data.

Enable it with:

```powershell
$env:PUBLIC_DEMO_MODE="true"
```

Then run:

```powershell
uvicorn classification_api:app --host 127.0.0.1 --port 8000
```

In public demo mode, the backend uses:

```text
data/demo_labels.csv
data/demo_articles.jsonl
runtime_state/demo_feedback.csv
```

It does not require:

```text
manual_ground_truth_with_GSE_links_REFRESHED.csv
pmc_gse_articles.jsonl
Mohammad_doi.csv
```

The UI is the same as the local full version, but the data are synthetic and small.
