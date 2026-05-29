## Problem Statement

The current project has a strong research pipeline for classifying biomedical papers as `Primary` or `Reuse`, but it is still fragmented as a deployable product. Curators need a single workflow that can:

- accept pasted article text, local identifiers, DOI, and batch files
- extract `GSE` / accession evidence automatically
- classify with an auditable evidence trail
- collect reviewer corrections
- feed corrections back into the gold standard and future retrieval/training
- expose a path to local plugin packaging and ChatGPT App packaging

## Solution

Build a deployable provenance-classification service around the current evidence extraction and `linear_model_plus_rag` stack. The product will include a local operator UI, a REST API, async batch processing, DOI/article resolution with cache, a feedback merge path for curation, and packaging scaffolds for a repo-local Codex plugin and a future ChatGPT App.

## User Stories

1. As a curator, I want to paste article text and receive `Primary/Reuse/Unclear` plus evidence, so that I can review one paper quickly.
2. As a curator, I want to enter a local `paper_id` or PMCID and classify the indexed article, so that I do not have to paste text manually.
3. As a curator, I want to enter a DOI and have the system fetch/cache article content when possible, so that DOI can become a practical intake path.
4. As a curator, I want the system to extract `GSE` and other accession IDs, so that I can compare the predicted provenance against the article’s cited datasets.
5. As a curator, I want to see structured provenance evidence roles, so that I can understand why the model predicted `Primary` or `Reuse`.
6. As a curator, I want to inspect retrieved labeled neighbors, so that I can evaluate whether RAG support is appropriate.
7. As a curator, I want to submit a corrected label with reviewer notes, so that misclassifications become reusable training evidence.
8. As a team lead, I want reviewer corrections stored separately before merge, so that the original gold standard remains auditable.
9. As a maintainer, I want a merge command that folds approved feedback into a new labeled CSV, so that retraining and RAG updates can use reviewed corrections.
10. As an operator, I want to upload a CSV/JSONL/JSON batch and get a job ID, so that large batches do not block the browser.
11. As an operator, I want to poll job status and download results as JSON or CSV, so that batch classification fits operational workflows.
12. As a developer, I want the product spec captured in one PRD, so that packaging and implementation decisions are aligned.
13. As a developer, I want deep modules for DOI resolution, async jobs, and feedback merge, so that API routes stay thin and testable.
14. As a developer, I want a repo-local plugin scaffold, so that the project can be surfaced consistently inside Codex.
15. As a developer, I want a ChatGPT App packaging plan with explicit tool contracts, so that the same classifier can be exposed through remote MCP later.
16. As a product owner, I want the system to default to the strongest audited deterministic model and escalate only when needed, so that quality and cost stay defensible.
17. As a reviewer, I want the local UI to show route recommendations such as `auto_accept`, `llm_review`, and `human_review`, so that I know where human attention is required.
18. As a developer, I want the DOI fetch path to cache fetched articles, so that repeated classification does not depend on repeated network calls.
19. As a maintainer, I want the system to tolerate missing feedback files and empty feedback tables, so that curation workflows fail safely.
20. As a future integrator, I want the REST API, plugin packaging, and ChatGPT App plan to share one domain contract, so that integration surfaces remain consistent.

## Implementation Decisions

- The production classifier remains the canonical decision engine and continues to use evidence extraction, structured provenance tagging, provenance-aware retrieval, and `linear_model_plus_rag`.
- DOI and remote article resolution are encapsulated in a dedicated resolver module with on-disk cache.
- The resolver first checks local cache, then attempts Europe PMC lookup by DOI or PMCID, then stores normalized article records that match the local article-index shape.
- Batch execution is moved into a background job manager that persists status plus downloadable JSON/CSV outputs per job.
- The synchronous upload endpoint remains available, but async upload becomes the operational default for larger files.
- Reviewer feedback continues to append to a separate CSV, but the saved rows now include route metadata and timestamp fields suitable for later merge.
- A dedicated merge command produces a new labeled CSV instead of overwriting the original gold-standard file in place.
- The merged labeled CSV stays schema-compatible with the current benchmark/training pipeline.
- CSV reading must accept multiple encodings because the current gold-standard file is not reliably UTF-8.
- The repo gains a repo-local plugin scaffold and marketplace entry, but these remain packaging artifacts around the existing service rather than replacing the service.
- The ChatGPT App direction uses a decoupled architecture: remote MCP tools expose classification operations, while a widget renders evidence/result state.
- Tool descriptions for the future ChatGPT App must follow action-oriented “Use this when…” phrasing, consistent with current OpenAI remote MCP guidance.
- The first packaging milestone is scaffold/documentation level rather than full public submission readiness, because HTTPS hosting, tunnel setup, and real MCP validation are separate operational steps.

## Testing Decisions

- Good tests here verify external behavior: identifier resolution, DOI cache fallback, batch job lifecycle, downloadable artifacts, and merge output shape.
- The deepest test targets should be the DOI/article resolver, feedback merge command, and batch job manager.
- API validation should confirm route wiring, response shape, and file-download behavior.
- The current project already relies heavily on end-to-end verification through real script runs; the new modules should follow the same approach first, then accumulate narrower tests later.
- Packaging artifacts should be validated structurally: manifest files present, JSON parseable, and local run instructions aligned with the current codebase.

## Out of Scope

- Full public ChatGPT App submission
- Production auth and multi-user access control
- Cloud queue infrastructure
- Automatic retraining triggered directly from feedback submission
- Non-Europe PMC DOI sources beyond the initial resolver fallback
- Human adjudication workflow beyond single-row feedback submission

## Further Notes

- No issue tracker is configured in this workspace, so this PRD is stored locally instead of being published to an issue tracker.
- The benchmark winner remains the operational default; LLM-assisted paths remain optional escalation layers.
- The same domain contract should be preserved across REST API, local UI, plugin packaging, and future ChatGPT App tooling.

