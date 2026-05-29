# v5.5 RAG-grounded reviewer UI

This version clarifies that RAG has two roles:

1. **RAG vote only**: a baseline comparator based on retrieved labeled neighbors.
2. **RAG context for LLM**: retrieved labeled neighbors are injected into the LLM reviewer prompt when an LLM reviewer is enabled.

The user can select multiple baseline models for comparison and then apply one optional LLM reviewer layer to every selected baseline path. High-confidence baseline decisions remain locked.

Recommended demo paths:

- Stable public demo: `Hybrid baseline` + `No LLM reviewer`.
- RAG-grounded LLM demo: `Hybrid baseline` + `LLM classifier, force` + `RAG context on`.
- Model comparison demo: `Linear only` + `Hybrid baseline` + `RAG vote only` with the same LLM reviewer layer.
