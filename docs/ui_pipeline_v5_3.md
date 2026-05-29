# v5.3 Pipeline UI

This version changes the model-selection UI from a flat multi-model chooser into a two-layer decision pipeline:

1. **Base decision model**
   - Linear only
   - Hybrid baseline
   - RAG vote only

2. **Optional LLM reviewer layer**
   - No LLM reviewer
   - Sentence reviewer, routed
   - Sentence reviewer, force
   - LLM classifier, routed
   - LLM classifier, force

The LLM reviewer is no longer presented as a peer baseline model. It runs after the selected baseline model and is displayed as an advisory/reviewer layer. High-confidence baseline locking still applies.

## Backend endpoint

The UI now calls:

```text
POST /review_pipeline
```

with:

```json
{
  "title": "...",
  "text": "...",
  "paper_id": "...",
  "base_mode": "hybrid_baseline",
  "reviewer_mode": "llm_sentence_judge_force"
}
```

The response includes:

- `pipeline.base_mode`
- `pipeline.reviewer_mode`
- `pipeline.base_result`
- `pipeline.reviewer_result`
- `pipeline.policy`
- `comparison_results`

The UI shows the composed final decision, plus clickable inspection cards for the base model, reviewer result, and pipeline final result.
