# Curating Public Feedback

Use this checklist before approving a feedback row for the RAG bank.

## Approve only if all are true

- The corrected label is `Primary` or `Reuse`.
- The correction is supported by an evidence sentence or identifiable article section.
- The evidence matches the project definition:
  - `Primary`: authors mainly generated the relevant dataset themselves.
  - `Reuse`: authors mainly reused public or external datasets.
- The row is not spam, duplicate noise, or unsupported opinion.
- The article identifier, title, or evidence is enough to trace the decision.

## Reject if any are true

- The user only says “wrong” without evidence.
- The corrected label conflicts with the supplied evidence.
- The pasted text is irrelevant.
- The row contains private or sensitive information that should not be used.
- The correction requires more checking than you can currently perform.

## Suggested review statuses

```text
pending       newly submitted, not checked yet
needs_check   plausible but needs manual verification
approved      accepted for RAG-bank export
rejected      not suitable for RAG-bank update
```
