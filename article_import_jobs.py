#!/usr/bin/env python3
"""Background jobs for fetching article text from DOI / PMCID lists."""

from __future__ import annotations

import csv
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from article_fetcher import ArticleResolver
from article_fetcher import ArticleResolverConfig
from article_fetcher import looks_like_doi
from article_fetcher import looks_like_pmcid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_identifier_text(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    matches: List[str] = []
    matches.extend(m.group(0) for m in re.finditer(r"(?i)\bPMC\d+\b", raw))
    matches.extend(m.group(0) for m in re.finditer(r"(?i)\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", raw))
    return _dedupe_keep_order(matches)


def _records_from_upload(filename: str, raw_bytes: bytes) -> List[Dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    decoded = raw_bytes.decode("utf-8-sig")
    if suffix == ".csv":
        return list(csv.DictReader(StringIO(decoded)))
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in decoded.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        obj = json.loads(decoded)
        if isinstance(obj, list):
            return [dict(x) for x in obj]
        raise ValueError("JSON upload must be a list of objects.")
    if suffix == ".txt":
        return [{"identifier": item} for item in parse_identifier_text(decoded)]
    raise ValueError("Only .csv, .jsonl, .json, and .txt fetch uploads are supported.")


def extract_identifiers_from_records(records: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for row in records:
        for key in ("identifier", "doi", "pmcid", "paper_id"):
            value = str((row or {}).get(key, "") or "").strip()
            if looks_like_doi(value) or looks_like_pmcid(value):
                out.append(value)
                break
    return _dedupe_keep_order(out)


def identifiers_from_upload(filename: str, raw_bytes: bytes) -> List[str]:
    return extract_identifiers_from_records(_records_from_upload(filename, raw_bytes))


def resolve_identifier_list(
    *,
    identifiers: List[str],
    output_jsonl: Path,
    cache_dir: Path = Path("cache/article_fetch"),
    sleep_seconds: float = 0.2,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    resolver = ArticleResolver(ArticleResolverConfig(cache_dir=cache_dir))
    requested = _dedupe_keep_order(identifiers)
    written = 0
    failed: List[str] = []
    gse_hits = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        total = len(requested)
        for idx, identifier in enumerate(requested, start=1):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Fetch job cancelled by user.")
            article = resolver.resolve(identifier)
            if article is None:
                failed.append(identifier)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                if progress_callback is not None:
                    progress_callback(idx, total)
                continue
            gse_ids = list(article.get("gse_ids") or [])
            if gse_ids:
                gse_hits += 1
            record = {
                "paper_id": str(article.get("paper_id", "") or identifier),
                "pmcid": str(article.get("pmcid", "")),
                "doi": str(article.get("doi", "")),
                "published_doi": str(article.get("doi", "")),
                "title": str(article.get("title", "")),
                "article_url": str(article.get("article_url", "")),
                "gse_ids": gse_ids,
                "accessions": ", ".join(gse_ids),
                "full_text": str(article.get("text", "")),
                "source_identifier": identifier,
                "source": str(article.get("source", "")),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if progress_callback is not None:
                progress_callback(idx, total)
        if progress_callback is not None and total == 0:
            progress_callback(0, 0)
    return {
        "output_jsonl": str(output_jsonl),
        "requested_identifiers": len(requested),
        "written": written,
        "failed": len(failed),
        "failed_identifiers": failed[:50],
        "gse_hit_articles": gse_hits,
    }


class ArticleFetchJobManager:
    def __init__(self, root_dir: Path, max_workers: int = 1):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="article-fetch")
        self.lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.futures: Dict[str, Future[Any]] = {}

    def create_job(
        self,
        *,
        identifiers: List[str],
        output_jsonl: Path,
        cache_dir: Path,
        sleep_seconds: float,
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
            "output_jsonl": str(output_jsonl),
            "requested_identifiers": len(_dedupe_keep_order(identifiers)),
            "total": len(_dedupe_keep_order(identifiers)),
            "processed": 0,
            "progress_percent": 0.0,
            "cache_dir": str(cache_dir),
            "job_dir": str(job_dir),
        }
        with self.lock:
            self.jobs[job_id] = status
            self.cancel_flags[job_id] = threading.Event()
        self._write_status(job_id)
        future = self.executor.submit(self._run_job, job_id, identifiers, output_jsonl, cache_dir, sleep_seconds)
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
        identifiers: List[str],
        output_jsonl: Path,
        cache_dir: Path,
        sleep_seconds: float,
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

            summary = resolve_identifier_list(
                identifiers=identifiers,
                output_jsonl=output_jsonl,
                cache_dir=cache_dir,
                sleep_seconds=sleep_seconds,
                progress_callback=_progress,
                cancel_check=_cancelled,
            )
            self._update_job(
                job_id,
                status="completed",
                completed_at=utc_now_iso(),
                processed=int(summary.get("requested_identifiers", 0)),
                total=int(summary.get("requested_identifiers", 0)),
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
