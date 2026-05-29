# UI update v5.4: multi-base comparison with optional shared LLM reviewer

The UI now allows selecting multiple base decision models at the same time.

- Base models: Linear only, Hybrid baseline, RAG vote only.
- Optional LLM reviewer: none, sentence reviewer routed/force, LLM classifier routed/force.

When an LLM reviewer is selected, it is applied after each selected base model.
This creates comparable paths such as:

- Linear only + LLM classifier, force
- Hybrid baseline + LLM classifier, force
- RAG vote only + LLM classifier, force

The high-confidence baseline lock still applies independently to each path.
