#!/usr/bin/env python3
"""Build a GEO/GSE-focused JSONL article subset from Mohammad_doi.csv.

This reproduces the useful path from the older notebook:
1. filter Mohammad rows to GEO
2. normalize PMCID and GSE
3. choose papers known to mention GSE-linked GEO records
4. fetch Europe PMC full text by PMCID
5. write JSONL in the same shape as the current local article index
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from article_fetcher import ArticleResolver
from article_fetcher import ArticleResolverConfig
from article_fetcher import looks_like_pmcid
import evidence_modeling as em


GSE_FUZZY = re.compile(r"\bGSE[\s:\-_]*([0-9]{3,})\b", re.I)


def norm_pmc(x: Any) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    return s if s.upper().startswith("PMC") else f"PMC{s.replace('PMC', '').strip()}"


def norm_gse(x: Any) -> str:
    m = GSE_FUZZY.search(str(x or ""))
    return f"GSE{m.group(1)}" if m else ""


def build_pmc_to_gse(mapping_csv: Path) -> Dict[str, List[str]]:
    df = em.read_csv_flex(mapping_csv)
    geo = df[df["repository"].astype(str).str.upper().eq("GEO")].copy()
    geo["pmcid"] = geo["pmc_ID"].apply(norm_pmc)
    geo["gse"] = geo["accession"].apply(norm_gse)
    geo = geo[(geo["pmcid"] != "") & (geo["gse"] != "")]
    return geo.groupby("pmcid")["gse"].apply(lambda s: sorted(set(x for x in s if x))).to_dict()


def build_existing_article_index(existing_jsonl_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if not existing_jsonl_path or not Path(existing_jsonl_path).exists():
        return index
    for rec in em.iter_jsonl(Path(existing_jsonl_path)):
        paper_id = str(rec.get("paper_id", "")).strip()
        if not paper_id:
            continue
        text = em.extract_text_fields(rec)
        index[paper_id] = {
            "paper_id": paper_id,
            "title": str(rec.get("title", "")),
            "article_url": str(rec.get("article_url", "")),
            "source": str(rec.get("source", "")),
            "gse_ids": rec.get("gse_ids") or [],
            "doi": str(rec.get("published_doi", "") or rec.get("doi", "")),
            "text": text,
        }
    return index


def load_labeled_pmcids(labeled_csv_path: Optional[Path]) -> List[str]:
    if labeled_csv_path is None or not Path(labeled_csv_path).exists():
        return []
    df = em.read_csv_flex(Path(labeled_csv_path))
    candidate_cols = [col for col in ("paper_id", "pmcid", "pmc_ID") if col in df.columns]
    if not candidate_cols:
        return []
    out: List[str] = []
    for col in candidate_cols:
        out.extend(norm_pmc(x) for x in df[col].tolist())
    return sorted(set(x for x in out if looks_like_pmcid(x)))


def write_geo_subset(
    *,
    mapping_csv: Path,
    output_jsonl: Path,
    sample_size: int = 0,
    batch_start: int = 0,
    cache_dir: Path = Path("cache/article_fetch"),
    sleep_seconds: float = 0.2,
    existing_jsonl_path: Optional[Path] = None,
    exclude_labeled_csv_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    pmc_to_gse = build_pmc_to_gse(mapping_csv)
    all_pmcids = sorted(pmcid for pmcid in pmc_to_gse.keys() if looks_like_pmcid(pmcid))
    excluded_labeled = set(load_labeled_pmcids(exclude_labeled_csv_path))
    pmcids = [pmcid for pmcid in all_pmcids if pmcid not in excluded_labeled]
    batch_start = max(0, int(batch_start))
    if sample_size and sample_size > 0:
        pmcids = pmcids[batch_start : batch_start + int(sample_size)]
    else:
        pmcids = pmcids[batch_start:]

    existing_index = build_existing_article_index(existing_jsonl_path)
    resolver = ArticleResolver(ArticleResolverConfig(cache_dir=cache_dir))
    written = 0
    failed = 0
    local_hits = 0
    remote_hits = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        total = len(pmcids)
        for idx, pmcid in enumerate(pmcids, start=1):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Import job cancelled by user.")
            article = existing_index.get(pmcid)
            if article:
                local_hits += 1
            else:
                article = resolver.resolve(pmcid)
                if article:
                    remote_hits += 1
            if not article:
                failed += 1
                if progress_callback is not None:
                    progress_callback(idx, total)
                continue
            record = {
                "paper_id": pmcid,
                "title": str(article.get("title", "")),
                "article_url": str(article.get("article_url", "")),
                "source": str(article.get("source", "")),
                "gse_ids": pmc_to_gse.get(pmcid, []),
                "accessions": ", ".join(pmc_to_gse.get(pmcid, [])),
                "full_text": str(article.get("text", "")),
                "published_doi": str(article.get("doi", "")),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if sleep_seconds > 0 and not existing_index.get(pmcid):
                time.sleep(sleep_seconds)
            if progress_callback is not None:
                progress_callback(idx, total)
        if progress_callback is not None and total == 0:
            progress_callback(0, 0)
    return {
        "output_jsonl": str(output_jsonl),
        "total_geo_pmcids": len(all_pmcids),
        "skipped_gold_standard": len(excluded_labeled),
        "remaining_after_exclusion": max(0, len(all_pmcids) - len(excluded_labeled)),
        "batch_start": batch_start,
        "batch_size": int(sample_size),
        "requested_pmcids": len(pmcids),
        "written": written,
        "failed": failed,
        "local_hits": local_hits,
        "remote_hits": remote_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GEO/GSE-focused article JSONL from Mohammad_doi.csv")
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=Path("mohammad_geo_articles.jsonl"))
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all remaining PMCIDs; otherwise fetch this batch size")
    parser.add_argument("--batch-start", type=int, default=0, help="Start offset after gold-standard exclusion")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/article_fetch"))
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--existing-jsonl-path", type=Path, default=Path("pmc_gse_articles.jsonl"))
    parser.add_argument("--exclude-labeled-csv-path", type=Path, default=Path("manual_ground_truth_with_GSE_links_REFRESHED.csv"))
    args = parser.parse_args()
    summary = write_geo_subset(
        mapping_csv=args.mapping_csv,
        output_jsonl=args.output_jsonl,
        sample_size=args.sample_size,
        batch_start=args.batch_start,
        cache_dir=args.cache_dir,
        sleep_seconds=args.sleep_seconds,
        existing_jsonl_path=args.existing_jsonl_path,
        exclude_labeled_csv_path=args.exclude_labeled_csv_path,
    )
    print(f"[ok] wrote {summary['written']} rows to {summary['output_jsonl']}")
    print(f"[ok] failed {summary['failed']} PMCID fetches")
    print(f"[ok] local hits {summary['local_hits']} | remote hits {summary['remote_hits']}")


if __name__ == "__main__":
    main()
