# UI model selector update v5.2

This revision keeps the v5 cleanup but restores visible model selection.

Changes:

- The stable baseline model cards are visible by default.
- Users can select one mode for a clean single-path demo or multiple modes for side-by-side comparison.
- LLM reviewer modes remain available in a compact expandable section.
- A selected-mode bar shows exactly which modes will run.
- A reset button returns the selector to the stable `hybrid_baseline` default.

Recommended public demo mode: `Hybrid baseline`.

Recommended local LLM test mode: select only `LLM classifier, force`, then verify that the high-confidence baseline lock is shown in the LLM audit panel.
