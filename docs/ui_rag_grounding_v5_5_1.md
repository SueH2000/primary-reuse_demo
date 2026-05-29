# UI v5.5.1: Hide LLM grounding neighbors when disabled

This update clarifies the distinction between internal RAG retrieval and LLM prompt grounding.

- Retrieved neighbors may still be computed internally for hybrid/RAG baselines.
- If **Send retrieved labeled examples to the LLM reviewer prompt** is unchecked, the right-side inspection panel hides the LLM grounding-neighbor table and LLM RAG prompt context.
- If enabled, the panel shows the neighbors that were sent into the LLM reviewer prompt.

Recommended default for demos: enable LLM grounding with top-k = 3.
