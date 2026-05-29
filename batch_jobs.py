#!/usr/bin/env python3
"""Async batch job execution for the local classification service."""

from __future__ import annotations

import csv
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from production_classifier import SUPPORTED_RUNTIME_MODES
from production_classifier import comparison_label_column


BATCH_RESULT_FIELDNAMES = [
    "paper_id",
    "title",
    "final_label",
    "final_source",
    "recommended_route",
    "recommended_route_reason",
    "linear_model_plus_rag",
    "linear_model_plus_rag_conf",
    "rag_vote",
    "rag_vote_margin",
    "gse_ids",
    "gse_urls",
    "accession_list",
    "main_decision_sentence",
    "main_decision_role",
    "llm_proposed_label",
    "llm_confidence",
    "llm_used_for_final",
    "changed_from_hybrid",
    "override_applied",
    "override_status",
    "selected_modes",
] + [comparison_label_column(mode) for mode in SUPPORTED_RUNTIME_MODES]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class BatchJobManager:
    """Run batch jobs in background threads and persist result artifacts."""

    def __init__(self, root_dir: Path, max_workers: int = 2):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch-classifier")
        self.lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.futures: Dict[str, Future[Any]] = {}

    def create_job(
        self,
        *,
        classifier: Any,
        records: List[Dict[str, Any]],
        filename: str,
        use_llm: bool,
        force_llm: bool,
        llm_strategy: str,
        compare_modes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "job_id": job_id,
            "filename": filename,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "queued",
            "count": len(records),
            "total": len(records),
            "processed": 0,
            "progress_percent": 0.0,
            "current_row": 0,
            "total_rows": len(records),
            "current_mode": "",
            "current_identifier": "",
            "use_llm": bool(use_llm),
            "force_llm": bool(force_llm),
            "llm_strategy": llm_strategy,
            "compare_modes": list(compare_modes or []),
            "job_dir": str(job_dir),
            "results_json_path": str(job_dir / "results.json"),
            "results_csv_path": str(job_dir / "results.csv"),
        }
        with self.lock:
            self.jobs[job_id] = status
            self.cancel_flags[job_id] = threading.Event()
        self._write_status(job_id)
        future = self.executor.submit(
            self._run_job,
            job_id,
            classifier,
            records,
            use_llm,
            force_llm,
            llm_strategy,
            list(compare_modes or []),
        )
        with self.lock:
            self.futures[job_id] = future
        return status

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            cached = self.jobs.get(job_id)
        if cached is not None:
            return dict(cached)
        status_path = self.root_dir / job_id / "status.json"
        if status_path.exists():
            return json.loads(status_path.read_text(encoding="utf-8"))
        return None

    def is_live_job(self, job_id: str) -> bool:
        with self.lock:
            future = self.futures.get(job_id)
            job = self.jobs.get(job_id)
        if future is None or job is None:
            return False
        return not future.done() and str(job.get("status", "")) in {"queued", "running", "cancelling"}

    def cancel_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(job_id)
            future = self.futures.get(job_id)
            cancel_flag = self.cancel_flags.get(job_id)
        if job is None:
            status_path = self.root_dir / job_id / "status.json"
            if not status_path.exists():
                return None
            persisted = json.loads(status_path.read_text(encoding="utf-8"))
            if str(persisted.get("status", "")) not in {"completed", "failed", "cancelled"}:
                persisted["status"] = "cancelled"
                persisted["completed_at"] = utc_now_iso()
                persisted["updated_at"] = utc_now_iso()
                persisted["cancel_note"] = "Cancelled from the UI after reconnect. No live worker was attached."
                status_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
            return persisted
        status = str(job.get("status", ""))
        if status in {"completed", "failed", "cancelled"}:
            return dict(job)
        if cancel_flag is not None:
            cancel_flag.set()
        if future is not None and future.cancel():
            self._update_job(job_id, status="cancelled", completed_at=utc_now_iso(), cancel_note="Cancelled before execution started.")
            return self.get_job(job_id)
        self._update_job(job_id, status="cancelling")
        return self.get_job(job_id)

    def _run_job(
        self,
        job_id: str,
        classifier: Any,
        records: List[Dict[str, Any]],
        use_llm: bool,
        force_llm: bool,
        llm_strategy: str,
        compare_modes: List[str],
    ) -> None:
        self._update_job(job_id, status="running")
        try:
            def _cancelled() -> bool:
                with self.lock:
                    flag = self.cancel_flags.get(job_id)
                return bool(flag and flag.is_set())

            def _progress(processed: int, total: int, detail: Optional[Dict[str, Any]] = None) -> None:
                percent = float(processed / total * 100.0) if total else 100.0
                payload: Dict[str, Any] = {
                    "processed": int(processed),
                    "count": int(total),
                    "progress_percent": round(percent, 1),
                }
                if detail:
                    payload.update(
                        {
                            "current_row": int(detail.get("current_row", 0) or 0),
                            "total_rows": int(detail.get("total_rows", len(records)) or len(records)),
                            "current_mode": str(detail.get("current_mode", "") or ""),
                            "current_identifier": str(detail.get("current_identifier", "") or ""),
                        }
                    )
                self._update_job(job_id, **payload)

            if compare_modes:
                results = classifier.classify_batch_selected_models(
                    records=records,
                    modes=compare_modes,
                    progress_callback=_progress,
                    cancel_check=_cancelled,
                )
            else:
                results = classifier.classify_batch_records(
                    records=records,
                    use_llm=use_llm,
                    force_llm=force_llm,
                    llm_strategy=llm_strategy,
                    progress_callback=_progress,
                    cancel_check=_cancelled,
                )
            status = self.get_job(job_id) or {}
            json_path = Path(status["results_json_path"])
            csv_path = Path(status["results_csv_path"])
            json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_csv(csv_path, results)
            self._update_job(
                job_id,
                status="completed",
                completed_at=utc_now_iso(),
                result_count=len(results),
                processed=len(results),
                progress_percent=100.0,
                label_summary=self._label_summary(results),
            )
        except Exception as exc:
            if "cancelled by user" in str(exc).lower():
                self._update_job(
                    job_id,
                    status="cancelled",
                    completed_at=utc_now_iso(),
                    cancel_note=str(exc),
                )
                return
            self._update_job(
                job_id,
                status="failed",
                completed_at=utc_now_iso(),
                error=str(exc),
            )

    def _write_status(self, job_id: str) -> None:
        status = self.get_job(job_id)
        if status is None:
            return
        path = self.root_dir / job_id / "status.json"
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            job = dict(self.jobs[job_id])
            job.update(fields)
            job["updated_at"] = utc_now_iso()
            self.jobs[job_id] = job
        self._write_status(job_id)

    def _label_summary(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for item in results:
            label = str((item.get("final") or {}).get("label", "missing"))
            summary[label] = summary.get(label, 0) + 1
        return summary

    def _write_csv(self, path: Path, results: List[Dict[str, Any]]) -> None:
        rows = [self._result_to_csv_row(item) for item in results]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BATCH_RESULT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def _result_to_csv_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        predictions = item.get("predictions") or {}
        evidence = item.get("evidence") or {}
        final = item.get("final") or {}
        audit = item.get("decision_audit") or {}
        comparison_results = item.get("comparison_results") or {}

        row = {
            "paper_id": item.get("paper_id") or (item.get("lookup") or {}).get("paper_id", ""),
            "title": item.get("title", ""),
            "final_label": final.get("label", ""),
            "final_source": final.get("source", ""),
            "recommended_route": item.get("recommended_route", ""),
            "recommended_route_reason": item.get("recommended_route_reason", ""),
            "linear_model_plus_rag": predictions.get("linear_model_plus_rag", ""),
            "linear_model_plus_rag_conf": predictions.get("linear_model_plus_rag_conf", ""),
            "rag_vote": predictions.get("rag_vote", ""),
            "rag_vote_margin": predictions.get("rag_vote_margin", ""),
            "gse_ids": ";".join(evidence.get("gse_ids", [])),
            "gse_urls": ";".join(evidence.get("gse_urls", [])),
            "accession_list": ";".join(evidence.get("accession_list", [])),
            "main_decision_sentence": evidence.get("main_decision_sentence", ""),
            "main_decision_role": evidence.get("main_decision_role", ""),
            "llm_proposed_label": audit.get("llm_proposed_label", ""),
            "llm_confidence": audit.get("llm_confidence", ""),
            "llm_used_for_final": audit.get("llm_used_for_final", ""),
            "changed_from_hybrid": audit.get("changed_from_hybrid", ""),
            "override_applied": audit.get("override_applied", ""),
            "override_status": audit.get("override_status", ""),
            "selected_modes": ";".join(item.get("selected_modes", [])),
        }
        for mode in SUPPORTED_RUNTIME_MODES:
            row[comparison_label_column(mode)] = ((comparison_results.get(mode) or {}).get("final") or {}).get("label", "")
        return row
