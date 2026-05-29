#!/usr/bin/env python3
"""FastAPI wrapper for the production classifier.

This is the easiest deployable surface for the current project:
- local web service
- future internal plugin/backend
- future ChatGPT app or external UI integration
"""

from __future__ import annotations

import os
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from article_import_jobs import ArticleFetchJobManager
from article_import_jobs import identifiers_from_upload
from article_import_jobs import parse_identifier_text
from batch_jobs import BatchJobManager
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic import Field
from starlette.staticfiles import StaticFiles

from import_mohammad_geo_subset import build_pmc_to_gse
import evidence_modeling as em
from merge_feedback_into_gold import merge_feedback
from production_classifier import DEFAULT_RUNTIME_MODES
from production_classifier import ProductionClassifier
from production_classifier import ProductionConfig
from production_classifier import SUPPORTED_LLM_STRATEGIES
from production_classifier import SUPPORTED_RUNTIME_MODES
from refresh_rag_bank import utc_now_iso
from supabase_feedback import build_pending_feedback_payload
from supabase_feedback import insert_pending_feedback
from supabase_feedback import supabase_feedback_enabled
from subset_import_jobs import MohammadImportJobManager


class ClassificationRequest(BaseModel):
    title: str = Field(default="", description="Paper title when available.")
    text: str = Field(description="Full article text, abstract+body, or pre-extracted article text.")
    paper_id: Optional[str] = None
    use_llm: bool = Field(default=False, description="If true, call the LLM when the row is routed to llm_review.")
    force_llm: bool = Field(default=False, description="If true, call the LLM even when routing would skip it.")
    llm_strategy: str = Field(
        default="classify",
        pattern="^(classify|verify_override|sentence_judge)$",
        description="classify = standard RAG+LLM final classifier, sentence_judge = let the LLM pick the key sentence and label from it, verify_override = reviewer mode.",
    )


class BatchClassificationRequest(BaseModel):
    items: List[ClassificationRequest]
    use_llm: bool = Field(default=False, description="Apply LLM escalation to routed rows in the batch.")
    force_llm: bool = Field(default=False, description="Force LLM on every batch row.")
    llm_strategy: str = Field(
        default="classify",
        pattern="^(classify|verify_override|sentence_judge)$",
        description="classify = standard RAG+LLM final classifier, sentence_judge = let the LLM pick the key sentence and label from it, verify_override = reviewer mode.",
    )


class IdentifierClassificationRequest(BaseModel):
    identifier: str = Field(description="Local identifier such as paper_id / PMC id / indexed identifier.")
    use_llm: bool = Field(default=False, description="If true, call the LLM when the row is routed to llm_review.")
    force_llm: bool = Field(default=False, description="If true, call the LLM even when routing would skip it.")
    llm_strategy: str = Field(
        default="classify",
        pattern="^(classify|verify_override|sentence_judge)$",
        description="classify = standard RAG+LLM final classifier, sentence_judge = let the LLM pick the key sentence and label from it, verify_override = reviewer mode.",
    )


class FeedbackRequest(BaseModel):
    paper_id: Optional[str] = None
    identifier: Optional[str] = None
    title: str = Field(default="", description="Optional title. If omitted and identifier resolves locally, title will be filled from the local article index.")
    text: str = Field(default="", description="Optional article text. If omitted and identifier resolves locally, text will be filled from the local article index.")
    predicted_label: str = Field(description="The model prediction shown to the reviewer.")
    corrected_label: str = Field(description="The reviewer-corrected final label.")
    reviewer: str = Field(default="", description="Reviewer or curator name.")
    reviewer_email: str = Field(default="", description="Optional reviewer email for follow-up. Stored only in feedback staging.")
    note: str = Field(default="", description="Optional correction note.")
    evidence_sentence: str = Field(default="", description="Optional reviewer-provided evidence sentence.")
    consent_to_store_input_text: bool = Field(default=False, description="If true, public feedback may store the submitted input text for curation.")
    result_json: Optional[Dict[str, Any]] = Field(default=None, description="Optional compact result JSON from the UI for audit context.")


class StrategyComparisonRequest(BaseModel):
    title: str = Field(default="", description="Paper title when available.")
    text: str = Field(default="", description="Article text when available.")
    paper_id: Optional[str] = None
    identifier: Optional[str] = None
    modes: Optional[List[str]] = Field(
        default=None,
        description="Subset of modes to compare: linear_only, hybrid_baseline, rag_vote_only, llm_sentence_judge_routed, llm_sentence_judge_force, llm_final_classify_routed, llm_final_classify_force.",
    )





class ReviewPipelineCompareRequest(BaseModel):
    title: str = Field(default="", description="Paper title when available.")
    text: str = Field(default="", description="Article text when available.")
    paper_id: Optional[str] = None
    identifier: Optional[str] = None
    base_modes: List[str] = Field(
        default_factory=lambda: ["hybrid_baseline"],
        description="One or more baseline decision models to compare: linear_only, hybrid_baseline, rag_vote_only.",
    )
    reviewer_mode: str = Field(
        default="none",
        pattern="^(none|llm_sentence_judge_routed|llm_sentence_judge_force|llm_final_classify_routed|llm_final_classify_force)$",
        description="Optional LLM reviewer layer applied after each selected base model.",
    )
    include_rag_context_for_llm: bool = Field(
        default=True,
        description="If true, retrieved labeled neighbors are injected into the LLM reviewer prompt.",
    )
    llm_rag_top_k: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional top-k retrieved neighbors to include in the LLM reviewer prompt. None uses classifier default.",
    )

class ReviewPipelineRequest(BaseModel):
    title: str = Field(default="", description="Paper title when available.")
    text: str = Field(default="", description="Article text when available.")
    paper_id: Optional[str] = None
    identifier: Optional[str] = None
    base_mode: str = Field(
        default="hybrid_baseline",
        pattern="^(linear_only|hybrid_baseline|rag_vote_only)$",
        description="Main baseline decision model.",
    )
    reviewer_mode: str = Field(
        default="none",
        pattern="^(none|llm_sentence_judge_routed|llm_sentence_judge_force|llm_final_classify_routed|llm_final_classify_force)$",
        description="Optional LLM reviewer layer to run after the base model.",
    )
    include_rag_context_for_llm: bool = Field(
        default=True,
        description="If true, retrieved labeled neighbors are injected into the LLM reviewer prompt.",
    )
    llm_rag_top_k: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional top-k retrieved neighbors to include in the LLM reviewer prompt. None uses classifier default.",
    )


def _validate_llm_strategy(llm_strategy: str) -> None:
    if llm_strategy not in SUPPORTED_LLM_STRATEGIES:
        allowed = ", ".join(SUPPORTED_LLM_STRATEGIES)
        raise HTTPException(status_code=400, detail=f"llm_strategy must be one of: {allowed}")


def _parse_selected_modes(raw_selected_models: str) -> List[str]:
    if not raw_selected_models.strip():
        return []
    parsed = [str(mode).strip() for mode in json.loads(raw_selected_models) if str(mode).strip()]
    invalid = [mode for mode in parsed if mode not in SUPPORTED_RUNTIME_MODES]
    if invalid:
        allowed = ", ".join(SUPPORTED_RUNTIME_MODES)
        raise HTTPException(status_code=400, detail=f"Unsupported selected model(s): {', '.join(invalid)}. Allowed: {allowed}")
    return parsed


class MohammadImportRequest(BaseModel):
    sample_size: int = Field(default=200, ge=0, description="Batch size. 0 means all remaining GEO/GSE-linked PMCIDs.")
    batch_start: int = Field(default=0, ge=0, description="Start offset after optional gold-standard exclusion.")
    skip_gold_standard: bool = Field(default=True, description="If true, skip PMID/PMCID rows that are already present in the labeled gold-standard CSV.")
    output_jsonl: str = Field(default="mohammad_geo_articles.jsonl", description="Output JSONL path. Relative paths are stored inside the import job folder.")


class ArticleSourceActivationRequest(BaseModel):
    jsonl_path: str = Field(description="JSONL file to use as the active extra lookup source.")


class RagBankRefreshRequest(BaseModel):
    output_csv: str = Field(default="rag_bank_refreshed.csv", description="Output CSV path for the refreshed human-reviewed labeled bank.")
    report_json: str = Field(default="rag_bank_refresh_report.json", description="JSON report path for the refresh summary.")
    activate: bool = Field(default=True, description="If true, make the refreshed labeled CSV the active classifier training/RAG bank immediately.")


def _env_path(name: str, default: str) -> Path:
    """Return a path from environment, falling back to a repository-local default."""
    return Path(os.getenv(name, default))


def _first_existing_path(*candidates: Path) -> Path:
    """Return the first existing path, or the first candidate if none exist.

    This keeps the same code usable in two modes:
    - local full mode with private training/article files in the project root
    - public demo mode with small safe files under data/
    """
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def is_public_demo_mode() -> bool:
    return os.getenv("PUBLIC_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


RUNTIME_STATE_DIR = Path("runtime_state")
ACTIVE_LOOKUP_SOURCE_PATH = RUNTIME_STATE_DIR / "active_lookup_source.json"
ACTIVE_LABELED_BANK_PATH = RUNTIME_STATE_DIR / "active_labeled_bank.json"


def get_default_jsonl_path() -> Path:
    env_value = os.getenv("CLASSIFIER_JSONL")
    if env_value:
        return Path(env_value)
    if is_public_demo_mode():
        return Path("data/demo_articles.jsonl")
    return _first_existing_path(Path("pmc_gse_articles.jsonl"), Path("data/demo_articles.jsonl"))


def get_base_labeled_csv_path() -> Path:
    env_value = os.getenv("CLASSIFIER_LABELED_CSV")
    if env_value:
        return Path(env_value)
    if is_public_demo_mode():
        return Path("data/demo_labels.csv")
    return _first_existing_path(Path("manual_ground_truth_with_GSE_links_REFRESHED.csv"), Path("data/demo_labels.csv"))


def get_default_mohammad_mapping_path() -> Path:
    env_value = os.getenv("CLASSIFIER_MOHAMMAD_MAPPING_CSV")
    if env_value:
        return Path(env_value)
    return Path("Mohammad_doi.csv")


def get_active_labeled_bank_path() -> Optional[Path]:
    if not ACTIVE_LABELED_BANK_PATH.exists():
        return None
    try:
        payload = json.loads(ACTIVE_LABELED_BANK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    path_value = str(payload.get("csv_path", "")).strip()
    if not path_value:
        return None
    return Path(path_value)


def get_active_lookup_source_path() -> Optional[Path]:
    if not ACTIVE_LOOKUP_SOURCE_PATH.exists():
        return None
    try:
        payload = json.loads(ACTIVE_LOOKUP_SOURCE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    path_value = str(payload.get("jsonl_path", "")).strip()
    if not path_value:
        return None
    return Path(path_value)


def set_active_lookup_source_path(jsonl_path: Path) -> None:
    RUNTIME_STATE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_LOOKUP_SOURCE_PATH.write_text(
        json.dumps({"jsonl_path": str(jsonl_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def set_active_labeled_bank_path(csv_path: Path) -> None:
    RUNTIME_STATE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_LABELED_BANK_PATH.write_text(
        json.dumps({"csv_path": str(csv_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@lru_cache(maxsize=1)
def get_classifier() -> ProductionClassifier:
    labeled_csv_path = get_active_labeled_bank_path() or get_base_labeled_csv_path()
    default_jsonl_path = get_default_jsonl_path()
    active_lookup_path = get_active_lookup_source_path()
    extra_lookup_paths: List[Path] = []
    if active_lookup_path is not None and active_lookup_path.exists() and active_lookup_path.resolve() != default_jsonl_path.resolve():
        extra_lookup_paths.append(active_lookup_path)
    cfg = ProductionConfig(
        labeled_csv_path=labeled_csv_path,
        jsonl_path=default_jsonl_path,
        extra_lookup_jsonl_paths=extra_lookup_paths,
        feedback_store_path=_env_path(
            "CLASSIFIER_FEEDBACK_STORE",
            "runtime_state/demo_feedback.csv" if is_public_demo_mode() else "rag_feedback_gold_standard.csv",
        ),
        mohammad_mapping_csv_path=get_default_mohammad_mapping_path(),
        extraction_mode=os.getenv("CLASSIFIER_EXTRACTION_MODE", "accession_windows"),
        win_before=int(os.getenv("CLASSIFIER_WIN_BEFORE", "350")),
        win_after=int(os.getenv("CLASSIFIER_WIN_AFTER", "900")),
        max_evidence_chars=int(os.getenv("CLASSIFIER_MAX_EVIDENCE_CHARS", "2200")),
        route_auto_accept_threshold=float(os.getenv("CLASSIFIER_ROUTE_AUTO_ACCEPT", "0.85")),
        route_llm_review_threshold=float(os.getenv("CLASSIFIER_ROUTE_LLM_REVIEW", "0.60")),
        ollama_url=os.getenv("CLASSIFIER_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")),
        ollama_model=os.getenv("CLASSIFIER_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3")),
        llm_override_lock_threshold=float(os.getenv("CLASSIFIER_LLM_OVERRIDE_LOCK_THRESHOLD", "0.85")),
        llm_num_predict=int(os.getenv("CLASSIFIER_LLM_NUM_PREDICT", "256")),
        llm_timeout_read=int(os.getenv("CLASSIFIER_LLM_TIMEOUT_READ", "300")),
    )
    return ProductionClassifier(cfg)


@lru_cache(maxsize=1)
def get_batch_job_manager() -> BatchJobManager:
    return BatchJobManager(root_dir=Path("batch_jobs"))


@lru_cache(maxsize=1)
def get_mohammad_import_job_manager() -> MohammadImportJobManager:
    return MohammadImportJobManager(root_dir=Path("import_jobs"))


@lru_cache(maxsize=1)
def get_article_fetch_job_manager() -> ArticleFetchJobManager:
    return ArticleFetchJobManager(root_dir=Path("fetch_jobs"))


def _read_job_statuses(root_dir: Path, kind: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = Path(root_dir)
    if not root.exists():
        return rows
    for child in root.iterdir():
        if not child.is_dir():
            continue
        status_path = child / "status.json"
        if not status_path.exists():
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["job_kind"] = kind
        rows.append(payload)
    rows.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return rows


def _annotate_server_jobs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    batch_manager = get_batch_job_manager()
    import_manager = get_mohammad_import_job_manager()
    fetch_manager = get_article_fetch_job_manager()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        kind = str(item.get("job_kind", ""))
        job_id = str(item.get("job_id", ""))
        if kind == "batch":
            live_managed = batch_manager.is_live_job(job_id)
        elif kind == "mohammad":
            live_managed = import_manager.is_live_job(job_id)
        elif kind == "fetch":
            live_managed = fetch_manager.is_live_job(job_id)
        else:
            live_managed = False
        item["live_managed"] = live_managed
        raw_status = str(item.get("status", ""))
        item["display_status"] = "stale" if raw_status in {"queued", "running", "cancelling"} and not live_managed else raw_status
        out.append(item)
    return out


app = FastAPI(
    title="Primary vs Reuse Classifier API",
    description="Production API for biomedical paper data-provenance classification.",
    version="0.1.0",
)

APP_DIR = Path(__file__).resolve().parent
WEBUI_DIR = APP_DIR / "webui"
if WEBUI_DIR.exists():
    app.mount("/ui-static", StaticFiles(directory=WEBUI_DIR), name="ui-static")


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    index_path = WEBUI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI assets were not found.")
    return FileResponse(index_path)


@app.get("/health")
def health() -> Dict[str, Any]:
    clf = get_classifier()
    meta = clf.metadata()
    return {
        "status": "ok",
        "public_demo_mode": is_public_demo_mode(),
        "using_demo_labeled_bank": get_base_labeled_csv_path().as_posix().startswith("data/"),
        "using_demo_article_index": get_default_jsonl_path().as_posix().startswith("data/"),
        "metadata": meta,
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return get_classifier().metadata()


@app.get("/capabilities")
def capabilities() -> Dict[str, Any]:
    return {
        "input_modes": [
            {
                "mode": "paste_text",
                "endpoint": "/classify",
                "required_fields": ["text"],
                "optional_fields": ["title", "paper_id", "use_llm", "force_llm", "llm_strategy"],
            },
            {
                "mode": "local_identifier_lookup",
                "endpoint": "/classify_identifier",
                "required_fields": ["identifier"],
                "notes": "Supports paper_id / PMCID lookup, local Mohammad DOI->PMCID mapping, and remote DOI/PMCID resolver fallback.",
            },
            {
                "mode": "strategy_comparison",
                "endpoint": "/compare_strategies",
                "notes": "Run any selected subset of non-LLM baselines and LLM routed/forced strategies side by side on the same article.",
                "supported_modes": list(SUPPORTED_RUNTIME_MODES),
            },
            {
                "mode": "batch_upload",
                "endpoint": "/classify_upload",
                "accepted_files": [".csv", ".jsonl", ".json"],
                "accepted_row_fields": ["text", "full_text", "article_text", "title", "paper_id", "pmcid", "doi", "identifier"],
            },
            {
                "mode": "batch_upload_async",
                "endpoint": "/classify_upload_async",
                "accepted_files": [".csv", ".jsonl", ".json"],
                "notes": "Recommended operational path for large files. Returns a job_id and downloadable results.",
            },
            {
                "mode": "feedback",
                "endpoint": "/feedback",
                "required_fields": ["predicted_label", "corrected_label"],
                "optional_fields": ["paper_id", "identifier", "title", "text", "reviewer", "note"],
            },
            {
                "mode": "refresh_rag_bank",
                "endpoint": "/refresh_rag_bank",
                "notes": "Public feedback is stored as pending review. Only curator-approved rows should be exported and merged into a refreshed labeled/RAG bank.",
            },
            {
                "mode": "import_mohammad_subset",
                "endpoint": "/import_mohammad_subset",
                "notes": "One-click GEO/GSE-focused import based on Mohammad_doi.csv while preserving the normal paste/identifier/upload flows.",
            },
            {
                "mode": "activate_lookup_source",
                "endpoint": "/article_source/activate",
                "notes": "Switch the active extra lookup JSONL without changing the base trained model.",
            },
            {
                "mode": "fetch_articles_async",
                "endpoint": "/fetch_articles_async",
                "accepted_inputs": ["textarea DOI/PMCID list", ".csv", ".jsonl", ".json", ".txt"],
                "notes": "Fetch new article text in batch from DOI / PMCID identifiers and write a reusable JSONL lookup source.",
            },
        ],
        "output_core_fields": [
            "final.label",
            "final.source",
            "recommended_route",
            "recommended_route_reason",
            "predictions.linear_model_plus_rag",
            "predictions.linear_model_plus_rag_conf",
            "predictions.rag_vote",
            "evidence.gse_ids",
            "evidence.gse_urls",
            "evidence.main_decision_gse_ids",
            "evidence.main_decision_gse_urls",
            "evidence.accession_list",
            "evidence.main_decision_sentence",
            "evidence.main_decision_role",
            "evidence.structured_evidence_summary",
            "rag.neighbors",
        ],
        "default_mohammad_mapping_csv": str(get_default_mohammad_mapping_path()),
    }


@app.post("/classify", summary="Classify one pasted article text")
def classify(req: ClassificationRequest) -> Dict[str, Any]:
    clf = get_classifier()
    return clf.classify(
        title=req.title,
        text=req.text,
        paper_id=req.paper_id,
        use_llm=req.use_llm,
        force_llm=req.force_llm,
        llm_strategy=req.llm_strategy,
    )


@app.post("/classify_batch", summary="Classify a JSON batch already loaded in the client")
def classify_batch(req: BatchClassificationRequest) -> Dict[str, Any]:
    clf = get_classifier()
    results = clf.classify_batch(
        items=[item.model_dump() for item in req.items],
        use_llm=req.use_llm,
        force_llm=req.force_llm,
        llm_strategy=req.llm_strategy,
    )
    return {"count": len(results), "results": results}


@app.post("/classify_identifier", summary="Classify one locally indexed identifier such as paper_id or PMC id")
def classify_identifier(req: IdentifierClassificationRequest) -> Dict[str, Any]:
    clf = get_classifier()
    return clf.classify_identifier(
        identifier=req.identifier,
        use_llm=req.use_llm,
        force_llm=req.force_llm,
        llm_strategy=req.llm_strategy,
    )


@app.post("/compare_strategies", summary="Compare baseline models and LLM strategies on the same paper")
def compare_strategies(req: StrategyComparisonRequest) -> Dict[str, Any]:
    clf = get_classifier()
    title = req.title or ""
    text = req.text or ""
    paper_id = req.paper_id
    identifier = req.identifier or ""
    requested_modes = req.modes or list(DEFAULT_RUNTIME_MODES)

    try:
        preferred = clf.run_selected_models(
            title=title,
            text=text,
            paper_id=paper_id,
            identifier=identifier,
            modes=requested_modes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "input": {
            "title": title,
            "paper_id": paper_id or "",
            "identifier": identifier,
            "has_text": bool(text.strip()),
        },
        "requested_modes": requested_modes,
        "results": preferred.get("comparison_results", {}),
        "preferred_mode": requested_modes[0] if requested_modes else "",
    }





def _inspection_card(result: Dict[str, Any], displayed_mode: str) -> Dict[str, Any]:
    """Return an acyclic, UI-friendly copy of a classification result."""
    return {
        "paper_id": result.get("paper_id", ""),
        "title": result.get("title", ""),
        "lookup": result.get("lookup", {}),
        "found": result.get("found", True),
        "message": result.get("message", ""),
        "final": result.get("final", {}),
        "predictions": result.get("predictions", {}),
        "recommended_route": result.get("recommended_route", ""),
        "recommended_route_reason": result.get("recommended_route_reason", ""),
        "evidence": result.get("evidence", {}),
        "rag": result.get("rag", {}),
        "llm": result.get("llm", {}),
        "decision_audit": result.get("decision_audit", {}),
        "pipeline": result.get("pipeline", {}),
        "displayed_mode": displayed_mode,
    }


@app.post("/review_pipeline_compare", summary="Compare one or more baseline models with the same optional LLM reviewer layer")
def review_pipeline_compare(req: ReviewPipelineCompareRequest) -> Dict[str, Any]:
    clf = get_classifier()
    allowed_base = {"linear_only", "hybrid_baseline", "rag_vote_only"}
    base_modes: List[str] = []
    for mode in req.base_modes or ["hybrid_baseline"]:
        clean = str(mode or "").strip()
        if not clean:
            continue
        if clean not in allowed_base:
            raise HTTPException(status_code=400, detail=f"Unsupported base mode: {clean}")
        if clean not in base_modes:
            base_modes.append(clean)
    if not base_modes:
        base_modes = ["hybrid_baseline"]

    results: Dict[str, Any] = {}
    raw_results: Dict[str, Any] = {}
    try:
        for base_mode in base_modes:
            result = clf.run_review_pipeline(
                title=req.title or "",
                text=req.text or "",
                paper_id=req.paper_id,
                identifier=req.identifier or "",
                base_mode=base_mode,
                reviewer_mode=req.reviewer_mode,
                include_rag_context_for_llm=req.include_rag_context_for_llm,
                llm_rag_top_k_override=req.llm_rag_top_k,
            )
            key = base_mode if req.reviewer_mode == "none" else f"{base_mode}__{req.reviewer_mode}"
            raw_results[key] = result
            results[key] = _inspection_card(result, key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    preferred_key = None
    for candidate in (
        ("hybrid_baseline" if req.reviewer_mode == "none" else f"hybrid_baseline__{req.reviewer_mode}"),
        next(iter(results.keys()), ""),
    ):
        if candidate in raw_results:
            preferred_key = candidate
            break
    if preferred_key is None:
        raise HTTPException(status_code=500, detail="No comparison result was produced.")

    preferred = _inspection_card(raw_results[preferred_key], preferred_key)
    preferred["comparison_results"] = results
    preferred["selected_modes"] = list(results.keys())
    preferred["displayed_mode"] = preferred_key
    preferred["pipeline_compare"] = {
        "base_modes": base_modes,
        "reviewer_mode": req.reviewer_mode,
        "include_rag_context_for_llm": bool(req.include_rag_context_for_llm),
        "llm_rag_top_k": req.llm_rag_top_k,
        "preferred_mode": preferred_key,
        "policy": "Compare selected baseline models side by side. When an LLM reviewer is enabled, retrieved RAG neighbors are injected into the reviewer prompt by default.",
    }
    return preferred

@app.post("/review_pipeline", summary="Run one baseline model plus an optional LLM reviewer layer")
def review_pipeline(req: ReviewPipelineRequest) -> Dict[str, Any]:
    clf = get_classifier()
    try:
        return clf.run_review_pipeline(
            title=req.title or "",
            text=req.text or "",
            paper_id=req.paper_id,
            identifier=req.identifier or "",
            base_mode=req.base_mode,
            reviewer_mode=req.reviewer_mode,
            include_rag_context_for_llm=req.include_rag_context_for_llm,
            llm_rag_top_k_override=req.llm_rag_top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _records_from_upload(filename: str, raw_bytes: bytes) -> List[Dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    decoded_utf8 = raw_bytes.decode("utf-8-sig")
    if suffix == ".csv":
        try:
            df = pd.read_csv(BytesIO(raw_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(raw_bytes), encoding="latin1")
        return df.fillna("").to_dict(orient="records")
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in decoded_utf8.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        obj = json.loads(decoded_utf8)
        if isinstance(obj, list):
            return [dict(x) for x in obj]
        raise HTTPException(status_code=400, detail="JSON upload must be a list of objects.")
    raise HTTPException(status_code=400, detail="Only .csv, .jsonl, and .json uploads are supported.")


def _fetch_identifiers_from_inputs(
    *,
    identifiers_text: str,
    file_name: str,
    raw_bytes: bytes,
) -> List[str]:
    file_identifiers: List[str] = []
    if raw_bytes:
        try:
            file_identifiers = identifiers_from_upload(file_name or "upload", raw_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    text_identifiers = parse_identifier_text(identifiers_text)
    merged = []
    seen = set()
    for value in text_identifiers + file_identifiers:
        key = str(value or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(key)
    return merged


@app.post("/classify_upload", summary="Upload CSV / JSONL / JSON and classify rows in batch")
async def classify_upload(
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
    force_llm: bool = Form(False),
    llm_strategy: str = Form("classify"),
    selected_models: str = Form(""),
) -> Dict[str, Any]:
    _validate_llm_strategy(llm_strategy)
    raw = await file.read()
    records = _records_from_upload(file.filename or "upload", raw)
    clf = get_classifier()
    compare_modes = _parse_selected_modes(selected_models)
    if compare_modes:
        results = clf.classify_batch_selected_models(records=records, modes=compare_modes)
    else:
        results = clf.classify_batch_records(
            records=records,
            use_llm=use_llm,
            force_llm=force_llm,
            llm_strategy=llm_strategy,
        )
    return {
        "filename": file.filename,
        "count": len(results),
        "results": results,
    }


@app.post("/classify_upload_async", summary="Upload CSV / JSONL / JSON and start an async batch job")
async def classify_upload_async(
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
    force_llm: bool = Form(False),
    llm_strategy: str = Form("classify"),
    selected_models: str = Form(""),
) -> Dict[str, Any]:
    _validate_llm_strategy(llm_strategy)
    raw = await file.read()
    records = _records_from_upload(file.filename or "upload", raw)
    manager = get_batch_job_manager()
    classifier = get_classifier()
    compare_modes = _parse_selected_modes(selected_models)
    job = manager.create_job(
        classifier=classifier,
        records=records,
        filename=file.filename or "upload",
        use_llm=use_llm,
        force_llm=force_llm,
        llm_strategy=llm_strategy,
        compare_modes=compare_modes,
    )
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "count": job["count"],
        "status_url": f"/jobs/{job['job_id']}",
        "download_json_url": f"/jobs/{job['job_id']}/download?format=json",
        "download_csv_url": f"/jobs/{job['job_id']}/download?format=csv",
    }


@app.get("/jobs/{job_id}", summary="Inspect async batch job status")
def get_job(job_id: str) -> Dict[str, Any]:
    job = get_batch_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/jobs/{job_id}/download", summary="Download async batch job artifacts")
def download_job(job_id: str, format: str = Query(default="json", pattern="^(json|csv)$")) -> FileResponse:
    job = get_batch_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if str(job.get("status")) != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed yet.")
    path = Path(job["results_json_path"] if format == "json" else job["results_csv_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result artifact not found.")
    media_type = "application/json" if format == "json" else "text/csv"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/server_jobs", summary="List recent server-side jobs so the browser can reconnect after refresh")
def server_jobs(limit: int = Query(default=10, ge=1, le=50)) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = []
    jobs.extend(_read_job_statuses(Path("batch_jobs"), "batch"))
    jobs.extend(_read_job_statuses(Path("import_jobs"), "mohammad"))
    jobs.extend(_read_job_statuses(Path("fetch_jobs"), "fetch"))
    jobs.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    jobs = _annotate_server_jobs(jobs)
    active = [row for row in jobs if bool(row.get("live_managed")) and str(row.get("status", "")) in {"queued", "running", "cancelling"}]
    return {
        "active_jobs": active[:limit],
        "recent_jobs": jobs[:limit],
    }


@app.post("/server_jobs/{job_kind}/{job_id}/cancel", summary="Cancel a server-side background job")
def cancel_server_job(job_kind: str, job_id: str) -> Dict[str, Any]:
    if job_kind == "batch":
        job = get_batch_job_manager().cancel_job(job_id)
    elif job_kind == "mohammad":
        job = get_mohammad_import_job_manager().cancel_job(job_id)
    elif job_kind == "fetch":
        job = get_article_fetch_job_manager().cancel_job(job_id)
    else:
        raise HTTPException(status_code=400, detail="job_kind must be batch, mohammad, or fetch.")
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    annotated = _annotate_server_jobs([job])[0]
    return {
        "cancelled": str(annotated.get("status", "")) in {"cancelled", "cancelling"},
        "job": annotated,
    }


@app.get("/mohammad_subset_info", summary="Inspect the default Mohammad GEO/GSE mapping source")
def mohammad_subset_info() -> Dict[str, Any]:
    mapping_path = get_default_mohammad_mapping_path()
    if not mapping_path.exists():
        return {
            "mapping_csv_path": str(mapping_path),
            "exists": False,
            "geo_pmcid_count": 0,
            "message": "Default Mohammad mapping CSV was not found.",
        }
    pmc_to_gse = build_pmc_to_gse(mapping_path)
    base_labeled_csv = get_base_labeled_csv_path()
    return {
        "mapping_csv_path": str(mapping_path),
        "exists": True,
        "geo_pmcid_count": len(pmc_to_gse),
        "gold_standard_csv_path": str(base_labeled_csv),
        "gold_standard_exists": base_labeled_csv.exists(),
        "example_pmcids": list(sorted(pmc_to_gse.keys())[:5]),
    }


@app.get("/article_source", summary="Inspect the current base and active lookup JSONL sources")
def article_source() -> Dict[str, Any]:
    base_path = get_default_jsonl_path()
    active_path = get_active_lookup_source_path()
    base_labeled_csv_path = get_base_labeled_csv_path()
    active_labeled_bank_path = get_active_labeled_bank_path()
    return {
        "base_jsonl_path": str(base_path),
        "base_exists": base_path.exists(),
        "active_lookup_jsonl_path": str(active_path) if active_path else "",
        "active_exists": bool(active_path and active_path.exists()),
        "base_labeled_csv_path": str(base_labeled_csv_path),
        "base_labeled_csv_exists": base_labeled_csv_path.exists(),
        "active_labeled_bank_path": str(active_labeled_bank_path) if active_labeled_bank_path else "",
        "active_labeled_bank_exists": bool(active_labeled_bank_path and active_labeled_bank_path.exists()),
    }


@app.post("/article_source/activate", summary="Activate a JSONL file as the current extra lookup source")
def activate_article_source(req: ArticleSourceActivationRequest) -> Dict[str, Any]:
    path = Path(req.jsonl_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"JSONL source not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise HTTPException(status_code=400, detail="Only .jsonl lookup sources can be activated.")
    try:
        first_record = next(em.iter_jsonl(path), None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse JSONL source: {exc}") from exc
    if first_record is None:
        raise HTTPException(status_code=400, detail="JSONL source is empty.")
    set_active_lookup_source_path(path.resolve())
    get_classifier.cache_clear()
    clf = get_classifier()
    return {
        "activated": True,
        "active_lookup_jsonl_path": str(path.resolve()),
        "indexed_articles": clf.metadata().get("indexed_articles"),
    }


@app.post("/import_mohammad_subset", summary="Start a one-click Mohammad GEO/GSE subset import job")
def import_mohammad_subset(req: MohammadImportRequest) -> Dict[str, Any]:
    mapping_path = get_default_mohammad_mapping_path()
    if not mapping_path.exists():
        raise HTTPException(status_code=404, detail=f"Mapping CSV not found: {mapping_path}")
    existing_jsonl_path = _env_path("CLASSIFIER_JSONL", "pmc_gse_articles.jsonl")
    job = get_mohammad_import_job_manager().create_job(
        mapping_csv=mapping_path,
        output_jsonl=Path(req.output_jsonl),
        sample_size=req.sample_size,
        batch_start=req.batch_start,
        cache_dir=Path("cache/article_fetch"),
        sleep_seconds=0.2,
        existing_jsonl_path=existing_jsonl_path if existing_jsonl_path.exists() else None,
        exclude_labeled_csv_path=get_base_labeled_csv_path() if req.skip_gold_standard else None,
    )
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "status_url": f"/import_jobs/{job['job_id']}",
        "download_jsonl_url": f"/import_jobs/{job['job_id']}/download",
        "activate_url": "/article_source/activate",
        "mapping_csv_path": str(mapping_path),
    }


@app.get("/import_jobs/{job_id}", summary="Inspect Mohammad subset import job status")
def get_import_job(job_id: str) -> Dict[str, Any]:
    job = get_mohammad_import_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Import job not found.")
    return job


@app.get("/import_jobs/{job_id}/download", summary="Download the imported Mohammad GEO/GSE JSONL artifact")
def download_import_job(job_id: str) -> FileResponse:
    job = get_mohammad_import_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Import job not found.")
    if str(job.get("status")) != "completed":
        raise HTTPException(status_code=409, detail="Import job is not completed yet.")
    path = Path(str(job.get("output_jsonl", "")))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Imported JSONL artifact not found.")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.post("/fetch_articles_async", summary="Fetch article text in batch from DOI / PMCID identifiers")
async def fetch_articles_async(
    identifiers_text: str = Form(""),
    output_jsonl: str = Form("fetched_articles.jsonl"),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    raw = await file.read() if file is not None else b""
    identifiers = _fetch_identifiers_from_inputs(
        identifiers_text=identifiers_text,
        file_name=file.filename if file is not None else "",
        raw_bytes=raw,
    )
    if not identifiers:
        raise HTTPException(status_code=400, detail="Provide at least one DOI or PMCID, either in the text box or an uploaded file.")
    job = get_article_fetch_job_manager().create_job(
        identifiers=identifiers,
        output_jsonl=Path(output_jsonl),
        cache_dir=Path("cache/article_fetch"),
        sleep_seconds=0.2,
    )
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "requested_identifiers": job["requested_identifiers"],
        "status_url": f"/fetch_jobs/{job['job_id']}",
        "download_jsonl_url": f"/fetch_jobs/{job['job_id']}/download",
        "activate_url": "/article_source/activate",
    }


@app.get("/fetch_jobs/{job_id}", summary="Inspect DOI / PMCID article fetch job status")
def get_fetch_job(job_id: str) -> Dict[str, Any]:
    job = get_article_fetch_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fetch job not found.")
    return job


@app.get("/fetch_jobs/{job_id}/download", summary="Download fetched DOI / PMCID article JSONL artifact")
def download_fetch_job(job_id: str) -> FileResponse:
    job = get_article_fetch_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fetch job not found.")
    if str(job.get("status")) != "completed":
        raise HTTPException(status_code=409, detail="Fetch job is not completed yet.")
    path = Path(str(job.get("output_jsonl", "")))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fetched JSONL artifact not found.")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.post("/feedback", summary="Store reviewer correction for curation before RAG-bank refresh")
def save_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    clf = get_classifier()
    local_result = clf.save_feedback(
        paper_id=req.paper_id,
        identifier=req.identifier,
        title=req.title,
        text=req.text,
        predicted_label=req.predicted_label,
        corrected_label=req.corrected_label,
        reviewer=req.reviewer,
        note=req.note,
    )

    # Public deployment path: store a pending review record in Supabase.
    # This does NOT update the RAG/gold-standard bank directly.
    if supabase_feedback_enabled():
        payload = build_pending_feedback_payload(
            row=dict(local_result.get("row") or {}),
            reviewer_email=req.reviewer_email,
            consent_to_store_input_text=req.consent_to_store_input_text,
            input_text=req.text,
            evidence_sentence=req.evidence_sentence,
            result_json=req.result_json or {},
        )
        try:
            supabase_result = insert_pending_feedback(payload)
            return {
                "saved": True,
                "storage": "supabase_pending_feedback",
                "review_status": "pending",
                "approved_for_rag": False,
                "message": "Thank you. Feedback was saved to the pending review database. It will not update the RAG bank until approved and exported.",
                "feedback_store_path": local_result.get("feedback_store_path", ""),
                "supabase": supabase_result,
                "row": local_result.get("row", {}),
            }
        except Exception as exc:
            # Keep the user informed instead of failing silently in the UI.
            # The local CSV fallback is useful during development, but it is not durable on free cloud hosts.
            return {
                "saved": True,
                "storage": "local_feedback_csv_supabase_failed",
                "review_status": "local_unreviewed",
                "approved_for_rag": False,
                "message": "Feedback was saved locally, but Supabase pending-feedback insert failed. Check Render environment variables, the rag_feedback table, and Supabase logs before relying on public collection.",
                "warning": str(exc),
                "feedback_store_path": local_result.get("feedback_store_path", ""),
                "row": local_result.get("row", {}),
            }

    return {
        **local_result,
        "storage": "local_feedback_csv",
        "review_status": "local_unreviewed",
        "approved_for_rag": False,
        "message": "Feedback saved locally. Review it before merging into a refreshed RAG bank.",
    }


@app.post("/refresh_rag_bank", summary="Merge reviewed feedback into a refreshed labeled/RAG bank and optionally activate it")
def refresh_rag_bank(req: RagBankRefreshRequest) -> Dict[str, Any]:
    base_csv = get_base_labeled_csv_path()
    feedback_csv = _env_path("CLASSIFIER_FEEDBACK_STORE", "rag_feedback_gold_standard.csv")
    output_csv = Path(req.output_csv)
    report_json = Path(req.report_json)

    if not base_csv.exists():
        raise HTTPException(status_code=404, detail=f"Base labeled CSV not found: {base_csv}")

    summary = merge_feedback(base_csv, feedback_csv, output_csv)
    summary.update(
        {
            "base_csv": str(base_csv),
            "feedback_csv": str(feedback_csv),
            "refreshed_at_utc": utc_now_iso(),
            "workflow_note": "Only reviewer-confirmed feedback rows are eligible for this refreshed bank because the feedback store is human-written.",
        }
    )
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    activated = False
    if req.activate:
        set_active_labeled_bank_path(output_csv.resolve())
        get_classifier.cache_clear()
        get_classifier()
        activated = True

    return {
        "refreshed": True,
        "activated": activated,
        "output_csv": str(output_csv.resolve()),
        "report_json": str(report_json.resolve()),
        "updated_rows": summary["updated_rows"],
        "appended_rows": summary["appended_rows"],
        "merged_rows": summary["merged_rows"],
        "feedback_exists": summary["feedback_exists"],
    }
