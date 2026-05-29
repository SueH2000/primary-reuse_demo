#!/usr/bin/env python3
"""Background jobs for importing the Mohammad GEO/GSE subset."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from import_mohammad_geo_subset import write_geo_subset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class MohammadImportJobManager:
    def __init__(self, root_dir: Path, max_workers: int = 1):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mohammad-import")
        self.lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.futures: Dict[str, Future[Any]] = {}

    def create_job(
        self,
        *,
        mapping_csv: Path,
        output_jsonl: Path,
        sample_size: int,
        batch_start: int,
        cache_dir: Path,
        sleep_seconds: float,
        existing_jsonl_path: Optional[Path],
        exclude_labeled_csv_path: Optional[Path],
    ) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl = output_jsonl if output_jsonl.is_absolute() else (job_dir / output_jsonl.name)
        status = {
            "job_id": job_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "queued",
            "total": 0,
            "processed": 0,
            "progress_percent": 0.0,
            "mapping_csv": str(mapping_csv),
            "output_jsonl": str(output_jsonl),
            "sample_size": int(sample_size),
            "batch_start": int(batch_start),
            "cache_dir": str(cache_dir),
            "existing_jsonl_path": str(existing_jsonl_path) if existing_jsonl_path else "",
            "exclude_labeled_csv_path": str(exclude_labeled_csv_path) if exclude_labeled_csv_path else "",
            "job_dir": str(job_dir),
        }
        with self.lock:
            self.jobs[job_id] = status
            self.cancel_flags[job_id] = threading.Event()
        self._write_status(job_id)
        future = self.executor.submit(
            self._run_job,
            job_id,
            mapping_csv,
            output_jsonl,
            sample_size,
            batch_start,
            cache_dir,
            sleep_seconds,
            existing_jsonl_path,
            exclude_labeled_csv_path,
        )
        with self.lock:
            self.futures[job_id] = future
        return status

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            cached = self.jobs.get(job_id)
        if cached is not None:
            return dict(cached)
        path = self.root_dir / job_id / "status.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
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
        mapping_csv: Path,
        output_jsonl: Path,
        sample_size: int,
        batch_start: int,
        cache_dir: Path,
        sleep_seconds: float,
        existing_jsonl_path: Optional[Path],
        exclude_labeled_csv_path: Optional[Path],
    ) -> None:
        self._update_job(job_id, status="running")
        try:
            def _cancelled() -> bool:
                with self.lock:
                    flag = self.cancel_flags.get(job_id)
                return bool(flag and flag.is_set())

            def _progress(processed: int, total: int) -> None:
                percent = float(processed / total * 100.0) if total else 100.0
                self._update_job(job_id, processed=int(processed), total=int(total), progress_percent=round(percent, 1))

            summary = write_geo_subset(
                mapping_csv=mapping_csv,
                output_jsonl=output_jsonl,
                sample_size=sample_size,
                batch_start=batch_start,
                cache_dir=cache_dir,
                sleep_seconds=sleep_seconds,
                existing_jsonl_path=existing_jsonl_path,
                exclude_labeled_csv_path=exclude_labeled_csv_path,
                progress_callback=_progress,
                cancel_check=_cancelled,
            )
            self._update_job(
                job_id,
                status="completed",
                completed_at=utc_now_iso(),
                processed=int(summary.get("requested_pmcids", 0)),
                total=int(summary.get("requested_pmcids", 0)),
                progress_percent=100.0,
                summary=summary,
                output_jsonl=str(output_jsonl),
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

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            job = dict(self.jobs[job_id])
            job.update(fields)
            job["updated_at"] = utc_now_iso()
            self.jobs[job_id] = job
        self._write_status(job_id)

    def _write_status(self, job_id: str) -> None:
        status = self.get_job(job_id)
        if status is None:
            return
        path = self.root_dir / job_id / "status.json"
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
