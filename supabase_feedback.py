#!/usr/bin/env python3
"""Supabase-backed staging store for public reviewer feedback.

Design principle:
- Public feedback is NEVER treated as gold standard immediately.
- Public feedback is inserted as pending review.
- Only curator-approved rows should be exported and merged into the refreshed RAG bank.

This module uses Supabase's PostgREST endpoint through requests, so it does not
require the optional supabase-py client package.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str
    table: str = "rag_feedback"
    timeout_seconds: int = 30

    @property
    def rest_url(self) -> str:
        return f"{self.url.rstrip('/')}/rest/v1/{self.table}"


def get_supabase_config() -> Optional[SupabaseConfig]:
    if not env_bool("USE_SUPABASE_FEEDBACK", False):
        return None
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "").strip()
    table = os.getenv("SUPABASE_FEEDBACK_TABLE", "rag_feedback").strip() or "rag_feedback"
    if not url or not key:
        return None
    return SupabaseConfig(url=url, key=key, table=table)


def supabase_feedback_enabled() -> bool:
    return get_supabase_config() is not None


def _headers(cfg: SupabaseConfig, prefer: str = "return=representation") -> Dict[str, str]:
    return {
        "apikey": cfg.key,
        "Authorization": f"Bearer {cfg.key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return json.loads(json.dumps(value, default=str))


def build_pending_feedback_payload(
    *,
    row: Dict[str, Any],
    reviewer_email: str = "",
    consent_to_store_input_text: bool = False,
    input_text: str = "",
    evidence_sentence: str = "",
    result_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert local feedback row into Supabase staging payload."""

    predicted = str(row.get("predicted_label", "") or "").strip()
    corrected = str(row.get("corrected_label", "") or "").strip()
    payload: Dict[str, Any] = {
        "paper_id": _string_or_none(row.get("paper_id")),
        "identifier": _string_or_none(row.get("identifier")),
        "title": _string_or_none(row.get("title")),
        "predicted_label": _string_or_none(predicted),
        "corrected_label": _string_or_none(corrected),
        "is_correct": predicted == corrected if predicted and corrected else None,
        "evidence_sentence": _string_or_none(evidence_sentence or row.get("main_decision_sentence")),
        "reviewer": _string_or_none(row.get("reviewer")),
        "reviewer_email": _string_or_none(reviewer_email),
        "reviewer_note": _string_or_none(row.get("note")),
        "input_text": input_text if consent_to_store_input_text else None,
        "result_json": _safe_json(result_json or {}),
        "review_status": "pending",
        "curator_note": None,
        "approved_for_rag": False,
        "local_feedback_row": _safe_json(row),
        "submitted_at_utc": utc_now_iso(),
        "source_app": "primary-reuse-public-demo",
        "consent_to_store_input_text": bool(consent_to_store_input_text),
    }
    return payload


def insert_pending_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_supabase_config()
    if cfg is None:
        raise RuntimeError("Supabase feedback is not configured. Set USE_SUPABASE_FEEDBACK=true, SUPABASE_URL, and SUPABASE_SERVICE_ROLE_KEY.")
    response = requests.post(
        cfg.rest_url,
        headers=_headers(cfg),
        data=json.dumps(payload, ensure_ascii=False),
        timeout=cfg.timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Supabase insert failed: HTTP {response.status_code} {response.text[:1000]}")
    try:
        body = response.json()
    except Exception:
        body = []
    inserted = body[0] if isinstance(body, list) and body else body
    return {
        "saved": True,
        "storage": "supabase_pending_feedback",
        "table": cfg.table,
        "inserted": inserted,
    }


def fetch_approved_feedback_rows(limit: int = 10000) -> List[Dict[str, Any]]:
    cfg = get_supabase_config()
    if cfg is None:
        raise RuntimeError("Supabase feedback is not configured.")
    params = {
        "select": "*",
        "review_status": "eq.approved",
        "approved_for_rag": "eq.true",
        "order": "created_at.asc",
        "limit": str(limit),
    }
    response = requests.get(
        cfg.rest_url,
        headers=_headers(cfg, prefer="return=minimal"),
        params=params,
        timeout=cfg.timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Supabase export failed: HTTP {response.status_code} {response.text[:1000]}")
    return response.json()


FEEDBACK_CSV_FIELDS = [
    "paper_id",
    "identifier",
    "title",
    "predicted_label",
    "corrected_label",
    "feedback_decision",
    "reviewer",
    "note",
    "timestamp_utc",
    "recommended_route",
    "recommended_route_reason",
    "pred_linear_model_plus_rag",
    "pred_linear_model_plus_rag_conf",
    "pred_rag_vote",
    "gse_ids",
    "gse_urls",
    "main_decision_gse_ids",
    "main_decision_gse_urls",
    "accession_list",
    "main_decision_sentence",
    "main_decision_role",
    "structured_evidence_summary",
    "evidence_text",
]


def approved_supabase_row_to_feedback_csv_row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map one approved Supabase row to merge_feedback_into_gold.py-compatible CSV row."""
    local = item.get("local_feedback_row") or {}
    if isinstance(local, str):
        try:
            local = json.loads(local)
        except Exception:
            local = {}
    row = {field: "" for field in FEEDBACK_CSV_FIELDS}
    for field in FEEDBACK_CSV_FIELDS:
        if field in local:
            row[field] = local.get(field, "")
    # Supabase-level fields override local row fields where relevant.
    row["paper_id"] = item.get("paper_id") or row["paper_id"]
    row["identifier"] = item.get("identifier") or row["identifier"]
    row["title"] = item.get("title") or row["title"]
    row["predicted_label"] = item.get("predicted_label") or row["predicted_label"]
    row["corrected_label"] = item.get("corrected_label") or row["corrected_label"]
    row["reviewer"] = item.get("reviewer") or row["reviewer"]
    row["note"] = item.get("reviewer_note") or row["note"]
    row["timestamp_utc"] = item.get("reviewed_at") or item.get("created_at") or item.get("submitted_at_utc") or row["timestamp_utc"]
    if not row["feedback_decision"]:
        row["feedback_decision"] = "confirmed_correct" if row["predicted_label"] == row["corrected_label"] else "corrected_label"
    return row


def write_feedback_csv(rows: Iterable[Dict[str, Any]], output_csv: Path) -> Dict[str, Any]:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_CSV_FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow({field: item.get(field, "") for field in FEEDBACK_CSV_FIELDS})
            count += 1
    return {"output_csv": str(output_csv), "rows": count}
