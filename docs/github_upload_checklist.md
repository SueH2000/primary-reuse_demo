# GitHub upload checklist

Before pushing:

```powershell
git status
```

Make sure these files are **not** staged:

```text
manual_ground_truth_with_GSE_links_REFRESHED.csv
pmc_gse_articles.jsonl
Mohammad_doi.csv
rag_feedback_gold_standard.csv
cache/
outputs*/
.venv/
```

First push:

```powershell
git init
git add .
git commit -m "Initial public provenance review console"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```
