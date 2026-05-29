# LLM Guardrails

The public demo treats `linear_model_plus_rag` as the stable decision path. LLM modes are reviewer aids, not final authority.

## Why this exists

In smoke tests, direct LLM classification can see correct RAG evidence but still misinterpret `downloaded/reanalyzed public GEO data` as `Primary` because the authors performed downstream analysis. For this task, downstream analysis of public data is still `Reuse` unless the authors generated the main dataset themselves.

## High-confidence baseline lock

If the hybrid baseline route is `auto_accept` and `linear_model_plus_rag_conf >= CLASSIFIER_LLM_OVERRIDE_LOCK_THRESHOLD`, the LLM may still run, but it cannot overwrite the final label.

Default threshold:

```text
CLASSIFIER_LLM_OVERRIDE_LOCK_THRESHOLD=0.85
```

Expected behavior:

```text
baseline = Reuse
baseline_confidence = 0.939
LLM proposed = Primary
final = Reuse
override_status = blocked_high_confidence_baseline
```

## Recommended demo mode

Use `Hybrid baseline` as the main public demo mode. Use LLM modes only for diagnostic comparison. Prefer `verify_override` for local experiments; treat `final classify force` as an experimental stress test.
