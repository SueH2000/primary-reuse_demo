#!/usr/bin/env python3
"""Resolve DOI/PMCID identifiers into cached article text records.

This module keeps remote lookup logic out of the API routes and classifier.
It prefers local cache, then Europe PMC metadata/full text, and falls back to
abstract-only records when full text is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests


DOI_PATTERN = re.compile(r"(?i)\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b")
GSE_PATTERN = re.compile(r"(?i)\bGSE\d+\b")


def normalize_doi(identifier: str) -> str:
    raw = (identifier or "").strip()
    raw = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", raw)
    raw = re.sub(r"(?i)^doi:\s*", "", raw)
    return raw.strip()


def looks_like_doi(identifier: str) -> bool:
    return bool(DOI_PATTERN.search(normalize_doi(identifier)))


def looks_like_pmcid(identifier: str) -> bool:
    return bool(re.fullmatch(r"(?i)PMC\d+", (identifier or "").strip()))


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass
class ArticleResolverConfig:
    cache_dir: Path = Path("cache/article_fetch")
    user_agent: str = "primary-vs-reuse-classifier/0.1"
    request_timeout: int = 30
    retry_delay_seconds: float = 0.8


class ArticleResolver:
    """Resolve identifiers to article records and cache the result on disk."""

    def __init__(self, cfg: Optional[ArticleResolverConfig] = None):
        self.cfg = cfg or ArticleResolverConfig()
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg.user_agent})

    def resolve(self, identifier: str) -> Optional[Dict[str, Any]]:
        key = (identifier or "").strip()
        if not key:
            return None
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        if looks_like_doi(key):
            record = self._resolve_doi(normalize_doi(key))
        elif looks_like_pmcid(key):
            record = self._resolve_pmcid(key.upper())
        else:
            record = None

        if record is not None:
            self._save_cache(key, record)
        return record

    def _cache_path(self, identifier: str) -> Path:
        digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
        return self.cfg.cache_dir / f"{digest}.json"

    def _load_cache(self, identifier: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(identifier)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, identifier: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(identifier)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_json(self, url: str) -> Optional[Dict[str, Any]]:
        for attempt in range(2):
            try:
                resp = self.session.get(url, timeout=self.cfg.request_timeout)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                pass
            if attempt == 0:
                time.sleep(self.cfg.retry_delay_seconds)
        return None

    def _get_text(self, url: str) -> Optional[str]:
        for attempt in range(2):
            try:
                resp = self.session.get(url, timeout=self.cfg.request_timeout)
                if resp.status_code == 200:
                    return resp.text
            except requests.RequestException:
                pass
            if attempt == 0:
                time.sleep(self.cfg.retry_delay_seconds)
        return None

    def _resolve_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        query = quote(f'DOI:"{doi}"')
        search_url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={query}&format=json&pageSize=1&resultType=core"
        )
        payload = self._get_json(search_url)
        result_list = (((payload or {}).get("resultList") or {}).get("result")) or []
        if not result_list:
            return None
        return self._build_record_from_europepmc(result_list[0], source_identifier=doi)

    def _resolve_pmcid(self, pmcid: str) -> Optional[Dict[str, Any]]:
        core_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        xml_text = self._get_text(core_url)
        if xml_text:
            record = self._record_from_fulltext_xml(pmcid=pmcid, xml_text=xml_text)
            if record is not None:
                return record
        meta_url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query=PMCID:{quote(pmcid)}&format=json&pageSize=1&resultType=core"
        )
        payload = self._get_json(meta_url)
        result_list = (((payload or {}).get("resultList") or {}).get("result")) or []
        if not result_list:
            return None
        return self._build_record_from_europepmc(result_list[0], source_identifier=pmcid)

    def _build_record_from_europepmc(self, row: Dict[str, Any], source_identifier: str) -> Optional[Dict[str, Any]]:
        pmcid = str(row.get("pmcid", "") or "").strip().upper()
        doi = normalize_doi(str(row.get("doi", "") or source_identifier))
        if pmcid:
            xml_text = self._get_text(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
            if xml_text:
                record = self._record_from_fulltext_xml(pmcid=pmcid, xml_text=xml_text, doi=doi, meta=row)
                if record is not None:
                    return record

        title = _collapse_whitespace(str(row.get("title", "")))
        abstract = _collapse_whitespace(str(row.get("abstractText", "")))
        article_text = "\n\n".join(part for part in [title, abstract] if part)
        if not article_text:
            return None
        return {
            "paper_id": pmcid or source_identifier,
            "pmcid": pmcid,
            "doi": doi,
            "title": title,
            "text": article_text,
            "gse_ids": sorted(set(m.group(0).upper() for m in GSE_PATTERN.finditer(article_text))),
            "article_url": f"https://europepmc.org/article/MED/{row.get('pmid')}" if row.get("pmid") else "",
            "source": "EuropePMC_abstract",
            "source_identifier": source_identifier,
        }

    def _record_from_fulltext_xml(
        self,
        *,
        pmcid: str,
        xml_text: str,
        doi: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None

        title = ""
        title_node = root.find(".//article-title")
        if title_node is not None:
            title = _collapse_whitespace("".join(title_node.itertext()))

        abstract_node = root.find(".//abstract")
        abstract = _collapse_whitespace(" ".join(abstract_node.itertext())) if abstract_node is not None else ""

        body_node = root.find(".//body")
        body = _collapse_whitespace(" ".join(body_node.itertext())) if body_node is not None else ""
        full_text = "\n\n".join(part for part in [title, abstract, body] if part)
        if not full_text:
            return None

        resolved_doi = doi
        if not resolved_doi:
            if meta is not None:
                resolved_doi = normalize_doi(str(meta.get("doi", "")))
            if not resolved_doi:
                for node in root.findall(".//article-id"):
                    id_type = str(node.attrib.get("pub-id-type", "")).lower()
                    if id_type == "doi":
                        resolved_doi = normalize_doi("".join(node.itertext()))
                        if resolved_doi:
                            break

        return {
            "paper_id": pmcid,
            "pmcid": pmcid,
            "doi": resolved_doi,
            "title": title,
            "text": full_text,
            "gse_ids": sorted(set(m.group(0).upper() for m in GSE_PATTERN.finditer(full_text))),
            "article_url": f"https://europepmc.org/article/PMC/{pmcid.removeprefix('PMC')}",
            "source": "EuropePMC_fullTextXML_remote",
        }

