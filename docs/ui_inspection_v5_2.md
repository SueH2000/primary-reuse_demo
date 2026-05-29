# UI update v5.2: model inspection without clutter

This version keeps model selection visible but makes the result panel easier to interpret.

## Changes

- Added `Displayed mode` to the result summary so users know which model's audit is currently shown.
- When multiple modes are selected, the comparison panel appears near the top of the result column.
- Comparison cards are now clickable. Click a card to inspect that mode's full audit, evidence, LLM status, and final label.
- The main result defaults to the last selected mode in multi-model comparison, so selecting a force LLM mode makes the LLM audit visible by default.
- Added a multi-mode notice explaining that the right panel shows one inspected mode at a time.
- Kept baseline model selection visible while leaving LLM modes in a compact advanced section.

## Demo recommendation

For a clean demo, select only `Hybrid baseline`.
For model comparison, select `Hybrid baseline` plus one LLM mode, then click the comparison cards to switch the full audit view.
