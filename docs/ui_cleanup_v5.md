# v5 UI cleanup

This version keeps the original FastAPI + static `webui/` console, but reduces visual noise in the default demo path.

## What changed

- The stable default mode is now visually separated from advanced comparison modes.
- The large execution-flow diagram is collapsed under **Advanced model comparison and LLM reviewer modes**.
- The right-side result panel now includes a decision-policy banner explaining whether the final label came from the hybrid baseline or LLM-assisted review.
- LLM audit now shows guardrail fields:
  - advisory-only status
  - override lock applied
  - override lock threshold
- Internal model signals and long evidence fields are still available, but less visually dominant.

## Intended demo behavior

For normal public demo use, keep **Hybrid baseline** selected. The LLM modes should be shown as experimental reviewer aids.

If an LLM mode is selected and the hybrid baseline is high-confidence, the final decision remains locked to the baseline while the LLM output is displayed as advisory information.
