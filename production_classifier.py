#!/usr/bin/env python3
"""Production-facing classifier built from the benchmarked evidence pipeline.

This module turns the current best-performing components into a reusable
inference service layer:
- evidence extraction
- structured provenance tagging
- provenance-aware RAG retrieval
- hybrid linear_model_plus_rag prediction
- optional LLM escalation

Unlike the benchmark script, production fitting uses all available labeled rows
to train one deployable model.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from article_fetcher import ArticleResolver
from article_fetcher import ArticleResolverConfig
import evidence_modeling as em

SUPPORTED_LLM_STRATEGIES = ("classify", "verify_override", "sentence_judge")
LLM_RUNTIME_MODE_SPECS = {
    "llm_sentence_judge_routed": {"use_llm": True, "force_llm": False, "llm_strategy": "sentence_judge"},
    "llm_sentence_judge_force": {"use_llm": True, "force_llm": True, "llm_strategy": "sentence_judge"},
    "llm_final_classify_routed": {"use_llm": True, "force_llm": False, "llm_strategy": "classify"},
    "llm_final_classify_force": {"use_llm": True, "force_llm": True, "llm_strategy": "classify"},
}
BASELINE_RUNTIME_MODES = ("linear_only", "hybrid_baseline", "rag_vote_only")
SUPPORTED_RUNTIME_MODES = BASELINE_RUNTIME_MODES + tuple(LLM_RUNTIME_MODE_SPECS.keys())
DEFAULT_RUNTIME_MODES = (
    "hybrid_baseline",
    "llm_sentence_judge_routed",
    "llm_final_classify_routed",
)


def gse_urls_from_ids(gse_ids: List[str]) -> List[str]:
    urls: List[str] = []
    for gse in gse_ids:
        gse_clean = str(gse).strip().upper()
        if gse_clean.startswith("GSE"):
            urls.append(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_clean}")
    return urls


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _sentence_candidates(text: str) -> List[str]:
    flat = _normalize_space(str(text or "").replace(" ... ", ". "))
    if not flat:
        return []
    parts = [p.strip() for p in em.SECTION_SPLIT.split(flat) if p and p.strip()]
    return [_normalize_space(p) for p in parts if _normalize_space(p)]


def _score_decision_sentence(sentence: str, target_label: str) -> float:
    text = _normalize_space(sentence)
    if not text:
        return -1.0
    score = 0.0
    if em.ACC_ANY.search(text):
        score += 2.5
    if em.GEO_WORDS.search(text):
        score += 2.0
    if em.PROV_WORDS.search(text):
        score += 1.5

    target = str(target_label or "").strip()
    if target == "Reuse":
        if em.DOWNLOAD_CUE_WORDS.search(text):
            score += 4.0
        if em.REANALYSIS_CUE_WORDS.search(text):
            score += 3.0
        if em.REUSE_CUE_WORDS.search(text):
            score += 2.0
        if em.PRIMARY_CUE_WORDS.search(text):
            score -= 2.0
        if em.DEPOSIT_CUE_WORDS.search(text) and em.WE_OUR.search(text):
            score -= 1.5
    elif target == "Primary":
        if em.PRIMARY_CUE_WORDS.search(text):
            score += 4.0
        if em.DEPOSIT_CUE_WORDS.search(text):
            score += 2.5
        if em.WE_OUR.search(text):
            score += 1.0
        if em.DOWNLOAD_CUE_WORDS.search(text):
            score -= 2.0
        if em.REANALYSIS_CUE_WORDS.search(text):
            score -= 1.5
    else:
        if em.PRIMARY_CUE_WORDS.search(text) or em.REUSE_CUE_WORDS.search(text):
            score += 1.0
    return score


def extract_main_decision_sentence(
    structured_items: List[Dict[str, Any]],
    evidence_text: str,
    target_label: str,
) -> Tuple[str, str]:
    ranked: List[Tuple[float, str, str]] = []
    label = str(target_label or "").strip()

    for item in structured_items or []:
        role = str(item.get("role", ""))
        coarse_label = str(item.get("coarse_label", ""))
        bonus = 0.0
        if coarse_label == label:
            bonus += 2.0
        if label == "Reuse" and role in {"Reuse_download", "Reuse_reanalysis"}:
            bonus += 1.5
        if label == "Primary" and role in {"Primary_generation", "Primary_deposition"}:
            bonus += 1.5
        for sentence in _sentence_candidates(str(item.get("text", ""))):
            ranked.append((_score_decision_sentence(sentence, label) + bonus, sentence, role))

    for sentence in _sentence_candidates(evidence_text):
        ranked.append((_score_decision_sentence(sentence, label), sentence, "evidence_text"))

    ranked = [item for item in ranked if item[1]]
    if not ranked:
        return "", ""
    ranked.sort(key=lambda item: (-item[0], -len(item[1])))
    best_score, best_sentence, best_role = ranked[0]
    if best_score <= 0:
        return best_sentence, best_role
    return best_sentence, best_role


def normalize_main_decision_role(role: str) -> str:
    value = str(role or "").strip()
    if value in {
        "Primary_generation",
        "Primary_deposition",
        "Reuse_download",
        "Reuse_reanalysis",
        "Mixed_conflict",
        "Other_provenance",
    }:
        return value
    return ""


PRIMARY_ROLES = {"Primary_generation", "Primary_deposition"}
REUSE_ROLES = {"Reuse_download", "Reuse_reanalysis"}


def extract_gse_ids(text: str) -> List[str]:
    return sorted(
        {
            str(m.group(0)).upper()
            for m in em.ACC_ANY.finditer(str(text or ""))
            if str(m.group(0)).upper().startswith("GSE")
        }
    )


def coerce_binary_primary_precedence(
    label: str,
    structured_items: List[Dict[str, Any]],
    primary_evidence: str = "",
    reuse_evidence: str = "",
) -> Tuple[str, str]:
    normalized = str(label or "").strip().title()
    role_set = {str(item.get("role", "")).strip() for item in structured_items or []}
    has_primary_role = bool(role_set & PRIMARY_ROLES)
    has_reuse_role = bool(role_set & REUSE_ROLES)
    has_primary_text = bool(em.PRIMARY_CUE_WORDS.search(str(primary_evidence or "")))
    has_reuse_text = bool(em.REUSE_CUE_WORDS.search(str(reuse_evidence or "")))

    if normalized == "Primary" or has_primary_role or has_primary_text:
        return "Primary", "primary_signal_present"
    if normalized == "Reuse" or has_reuse_role or has_reuse_text:
        return "Reuse", "reuse_signal_present"
    return "Reuse", "binary_default_to_reuse"


def choose_main_decision_gse(
    *,
    main_decision_sentence: str,
    main_decision_role: str,
    final_label: str,
    structured_items: List[Dict[str, Any]],
    all_gse_ids: List[str],
) -> List[str]:
    sentence_gse_ids = extract_gse_ids(main_decision_sentence)
    if sentence_gse_ids:
        return sentence_gse_ids

    preferred_roles = list(PRIMARY_ROLES if final_label == "Primary" else REUSE_ROLES)
    ranked_roles = [main_decision_role] + preferred_roles + ["Mixed_conflict", "Other_provenance"]
    seen_roles: List[str] = []
    for role in ranked_roles:
        role_value = str(role or "").strip()
        if role_value and role_value not in seen_roles:
            seen_roles.append(role_value)

    for role in seen_roles:
        for item in structured_items or []:
            if str(item.get("role", "")).strip() != role:
                continue
            item_gse_ids = [acc for acc in item.get("accessions", []) if str(acc).upper().startswith("GSE")]
            if item_gse_ids:
                return sorted({str(acc).upper() for acc in item_gse_ids})

    return list(all_gse_ids[:1])


def comparison_label_column(mode: str) -> str:
    return f"{mode}_label"


@dataclass
class ProductionConfig:
    labeled_csv_path: Path
    jsonl_path: Optional[Path] = None
    extra_lookup_jsonl_paths: Optional[List[Path]] = None
    feedback_store_path: Path = Path("rag_feedback_gold_standard.csv")
    article_cache_dir: Path = Path("cache/article_fetch")
    mohammad_mapping_csv_path: Optional[Path] = None
    extraction_mode: str = "accession_windows"
    win_before: int = 350
    win_after: int = 900
    max_evidence_chars: int = 2200
    max_features: int = 20000
    rag_top_k: int = 8
    rag_candidate_pool: int = 12
    rag_per_label_cap: int = 4
    rag_min_similarity: float = 0.01
    rag_vote_margin: float = 0.0
    llm_rag_top_k: int = 3
    route_auto_accept_threshold: float = 0.85
    route_llm_review_threshold: float = 0.60
    llm_hard_margin: float = 0.05
    # Guardrail: LLM output is advisory only for high-confidence auto-accepted baseline rows.
    # This prevents direct LLM classification from overwriting strong deterministic evidence.
    llm_override_lock_threshold: float = 0.85
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3"
    llm_temperature: float = 0.0
    llm_num_predict: int = 256
    llm_timeout_connect: int = 10
    llm_timeout_read: int = 300


class ProductionClassifier:
    """Reusable deployable classifier wrapping the benchmarked pipeline."""

    def __init__(self, cfg: ProductionConfig):
        self.cfg = cfg
        labeled_csv_path = Path(cfg.labeled_csv_path)
        jsonl_path = Path(cfg.jsonl_path) if cfg.jsonl_path else None
        self.model_cfg = em.ModelingConfig(
            labeled_csv_path=labeled_csv_path,
            jsonl_path=jsonl_path,
            out_dir=Path("outputs_production_runtime"),
            extraction_mode=cfg.extraction_mode,
            win_before=cfg.win_before,
            win_after=cfg.win_after,
            max_evidence_chars=cfg.max_evidence_chars,
            max_features=cfg.max_features,
            rag_top_k=cfg.rag_top_k,
            rag_candidate_pool=cfg.rag_candidate_pool,
            rag_per_label_cap=cfg.rag_per_label_cap,
            rag_min_similarity=cfg.rag_min_similarity,
            rag_vote_margin=cfg.rag_vote_margin,
            llm_hard_margin=cfg.llm_hard_margin,
            llm_rag_top_k=cfg.llm_rag_top_k,
            ollama_url=cfg.ollama_url,
            ollama_model=cfg.ollama_model,
            llm_temperature=cfg.llm_temperature,
            llm_num_predict=cfg.llm_num_predict,
            llm_timeout_connect=cfg.llm_timeout_connect,
            llm_timeout_read=cfg.llm_timeout_read,
            route_auto_accept_threshold=cfg.route_auto_accept_threshold,
            route_llm_review_threshold=cfg.route_llm_review_threshold,
        )
        self.dataset = em.build_labeled_dataset(self.model_cfg)
        if self.dataset.empty:
            raise ValueError("No labeled rows were available to build the production classifier.")
        self.article_index = self._build_article_index(jsonl_path)
        for extra_path in cfg.extra_lookup_jsonl_paths or []:
            self._merge_article_index(extra_path)
        self.feedback_store_path = Path(cfg.feedback_store_path)
        self.resolver = ArticleResolver(ArticleResolverConfig(cache_dir=Path(cfg.article_cache_dir)))
        self._register_mohammad_aliases(cfg.mohammad_mapping_csv_path)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_features=cfg.max_features,
        )
        X_text = self.vectorizer.fit_transform(self.dataset["evidence_text"].fillna(""))

        self.linear_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        self.linear_clf.fit(X_text, self.dataset["label"])

        self.retriever = em.RagExampleRetriever(self.dataset, max_features=cfg.max_features)
        train_rag_outputs = self.dataset.apply(
            lambda row: em.rag_vote_predict_with_details(
                title=str(row.get("title", "")),
                evidence_text=str(row.get("evidence_text", "")),
                retriever=self.retriever,
                top_k=cfg.rag_top_k,
                candidate_pool=cfg.rag_candidate_pool,
                per_label_cap=cfg.rag_per_label_cap,
                min_similarity=cfg.rag_min_similarity,
                vote_margin=cfg.rag_vote_margin,
                exclude_paper_id=str(row.get("paper_id", "")),
            ),
            axis=1,
        )
        train_profiles = self.dataset["evidence_text"].fillna("").astype(str).apply(em.provenance_profile)
        train_rag_dense = np.asarray(
            [em.rag_feature_row(result, profile) for result, profile in zip(train_rag_outputs, train_profiles)],
            dtype=float,
        )
        X_hybrid = hstack([X_text, csr_matrix(train_rag_dense)], format="csr")

        self.hybrid_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        self.hybrid_clf.fit(X_hybrid, self.dataset["label"])

        self.mined_phrases = em.mine_top_phrases(
            texts=self.dataset["evidence_text"],
            labels=self.dataset["label"],
            top_k=self.model_cfg.top_k_phrases,
            ngram_range=(self.model_cfg.ngram_min, self.model_cfg.ngram_max),
            min_df=self.model_cfg.min_df,
            max_features=self.model_cfg.max_features,
        )
        self.phrase_hints = em._mined_phrase_hints(self.mined_phrases, per_class=8)

    def _build_article_index(self, jsonl_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        if jsonl_path is None or not jsonl_path.exists():
            return index
        for rec in em.iter_jsonl(jsonl_path):
            self._register_article_from_record(index, rec)
        return index

    def _register_article_from_record(self, index: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> None:
        paper_id = str(rec.get("paper_id", "")).strip()
        if not paper_id:
            return
        article_text = em.extract_text_fields(rec)
        title = str(rec.get("title", "")).strip()
        gse_ids = rec.get("gse_ids") or []
        if isinstance(gse_ids, str):
            gse_ids = [x.strip() for x in gse_ids.split(";") if x.strip()]
        doi = str(rec.get("published_doi", "") or rec.get("doi", "")).strip()
        article = {
            "paper_id": paper_id,
            "title": title,
            "text": article_text,
            "gse_ids": list(gse_ids),
            "article_url": str(rec.get("article_url", "")),
            "source": str(rec.get("source", "")),
            "doi": doi,
            "pmcid": paper_id if paper_id.upper().startswith("PMC") else "",
        }
        for key in {
            paper_id,
            paper_id.upper(),
            paper_id.lower(),
            str(rec.get("article_url", "")).strip(),
            doi,
            doi.lower(),
        }:
            if key:
                index[key] = article

    def _merge_article_index(self, jsonl_path: Optional[Path]) -> None:
        if jsonl_path is None or not Path(jsonl_path).exists():
            return
        for rec in em.iter_jsonl(Path(jsonl_path)):
            self._register_article_from_record(self.article_index, rec)

    def _register_mohammad_aliases(self, mapping_csv_path: Optional[Path]) -> None:
        if not mapping_csv_path:
            return
        path = Path(mapping_csv_path)
        if not path.exists():
            return
        try:
            mapping_df = em.read_csv_flex(path)
        except Exception:
            return
        if "published_doi" not in mapping_df.columns or "pmc_ID" not in mapping_df.columns:
            return

        def _norm_pmc(x: Any) -> str:
            s = str(x or "").strip()
            if not s:
                return ""
            return s if s.upper().startswith("PMC") else f"PMC{s.replace('PMC', '').strip()}"

        for _, row in mapping_df.iterrows():
            doi = str(row.get("published_doi", "") or "").strip()
            pmcid = _norm_pmc(row.get("pmc_ID", ""))
            if not doi or not pmcid:
                continue
            article = self.article_index.get(pmcid) or self.article_index.get(pmcid.upper()) or self.article_index.get(pmcid.lower())
            if article is None:
                continue
            self.article_index[doi] = article
            self.article_index[doi.lower()] = article

    def metadata(self) -> Dict[str, Any]:
        label_counts = self.dataset["label"].value_counts().to_dict()
        return {
            "training_rows": int(len(self.dataset)),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "label_mode": "binary_primary_precedence",
            "label_rule": "If any core Primary signal is present, classify as Primary; otherwise classify as Reuse.",
            "labeled_csv_path": str(self.cfg.labeled_csv_path),
            "extraction_mode": self.cfg.extraction_mode,
            "route_auto_accept_threshold": float(self.cfg.route_auto_accept_threshold),
            "route_llm_review_threshold": float(self.cfg.route_llm_review_threshold),
            "rag_top_k": int(self.cfg.rag_top_k),
            "ollama_model": self.cfg.ollama_model,
            "indexed_articles": int(len({v["paper_id"] for v in self.article_index.values()})),
            "feedback_store_path": str(self.feedback_store_path),
            "article_cache_dir": str(self.cfg.article_cache_dir),
            "mohammad_mapping_csv_path": str(self.cfg.mohammad_mapping_csv_path or ""),
            "extra_lookup_jsonl_paths": [str(p) for p in (self.cfg.extra_lookup_jsonl_paths or [])],
        }

    def lookup_article(self, identifier: str) -> Optional[Dict[str, Any]]:
        if not identifier:
            return None
        key = str(identifier).strip()
        if not key:
            return None
        local = self.article_index.get(key) or self.article_index.get(key.upper()) or self.article_index.get(key.lower())
        if local is not None:
            return local
        resolved = self.resolver.resolve(key)
        if resolved is not None:
            self._register_article_aliases(resolved)
        return resolved

    def _register_article_aliases(self, article: Dict[str, Any]) -> None:
        paper_id = str(article.get("paper_id", "")).strip()
        doi = str(article.get("doi", "")).strip()
        pmcid = str(article.get("pmcid", "")).strip()
        url = str(article.get("article_url", "")).strip()
        for alias in {paper_id, paper_id.upper(), paper_id.lower(), doi, doi.lower(), pmcid, pmcid.upper(), pmcid.lower(), url}:
            if alias:
                self.article_index[alias] = article

    def _build_single_row(self, title: str, text: str, paper_id: Optional[str] = None) -> pd.DataFrame:
        evidence_text = em.build_evidence_text(
            text,
            extraction_mode=self.cfg.extraction_mode,
            max_chars=self.cfg.max_evidence_chars,
            win_before=self.cfg.win_before,
            win_after=self.cfg.win_after,
        )
        row_df = pd.DataFrame(
            [
                {
                    "paper_id": paper_id or "",
                    "title": title or "",
                    "article_text": text or "",
                    "evidence_text": evidence_text,
                }
            ]
        )
        return em.enrich_with_structured_evidence(row_df)

    def _serialize_neighbors(self, neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in neighbors:
            rows.append(
                {
                    "paper_id": str(item.get("paper_id", "")),
                    "label": str(item.get("label", "")),
                    "score": float(item.get("score", 0.0)),
                    "base_score": float(item.get("base_score", 0.0)),
                    "title": str(item.get("title", "")),
                    "structured_summary": str(item.get("structured_summary", "")),
                    "snippet": str(item.get("snippet", "")),
                }
            )
        return rows

    def _classify_selected_mode(
        self,
        *,
        mode: str,
        title: str = "",
        text: str = "",
        paper_id: Optional[str] = None,
        identifier: str = "",
        include_rag_context_for_llm: bool = True,
        llm_rag_top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        def _run_entry(**kwargs: Any) -> Dict[str, Any]:
            if text.strip():
                return self.classify(
                    title=title,
                    text=text,
                    paper_id=paper_id,
                    include_rag_context_for_llm=include_rag_context_for_llm,
                    llm_rag_top_k_override=llm_rag_top_k_override,
                    **kwargs,
                )
            return self.classify_identifier(
                identifier=identifier,
                include_rag_context_for_llm=include_rag_context_for_llm,
                llm_rag_top_k_override=llm_rag_top_k_override,
                **kwargs,
            )

        if mode == "linear_only":
            result = _run_entry(
                use_llm=False,
                force_llm=False,
                llm_strategy="classify",
            )
            result["final"] = {
                "label": str((result.get("predictions") or {}).get("linear_model", "")),
                "source": "linear_model",
            }
            return result
        if mode == "hybrid_baseline":
            return _run_entry(
                use_llm=False,
                force_llm=False,
                llm_strategy="classify",
            )
        if mode == "rag_vote_only":
            result = _run_entry(
                use_llm=False,
                force_llm=False,
                llm_strategy="classify",
            )
            result["final"] = {
                "label": str((result.get("predictions") or {}).get("rag_vote", "")),
                "source": "rag_vote",
            }
            return result
        return _run_entry(**LLM_RUNTIME_MODE_SPECS[mode])

    def classify(
        self,
        title: str,
        text: str,
        paper_id: Optional[str] = None,
        use_llm: bool = False,
        force_llm: bool = False,
        llm_strategy: str = "classify",
        include_rag_context_for_llm: bool = True,
        llm_rag_top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        row_df = self._build_single_row(title=title, text=text, paper_id=paper_id)
        row = row_df.iloc[0]
        evidence_text = str(row.get("evidence_text", ""))
        structured_summary = str(row.get("structured_evidence_summary", ""))
        structured_json = json.loads(str(row.get("structured_evidence_json", "[]")))

        rag_result = em.rag_vote_predict_with_details(
            title=str(row.get("title", "")),
            evidence_text=evidence_text,
            retriever=self.retriever,
            top_k=self.cfg.rag_top_k,
            candidate_pool=self.cfg.rag_candidate_pool,
            per_label_cap=self.cfg.rag_per_label_cap,
            min_similarity=self.cfg.rag_min_similarity,
            vote_margin=self.cfg.rag_vote_margin,
        )
        profile = em.provenance_profile(evidence_text)
        X_text = self.vectorizer.transform([evidence_text])
        rag_dense = np.asarray([em.rag_feature_row(rag_result, profile)], dtype=float)
        X_hybrid = hstack([X_text, csr_matrix(rag_dense)], format="csr")

        linear_label = str(self.linear_clf.predict(X_text)[0])
        linear_probs = em._probability_frame(self.linear_clf, X_text).iloc[0]
        hybrid_label = str(self.hybrid_clf.predict(X_hybrid)[0])
        hybrid_probs = em._probability_frame(self.hybrid_clf, X_hybrid).iloc[0]
        hybrid_conf = float(max(float(hybrid_probs["prob_Primary"]), float(hybrid_probs["prob_Reuse"])))

        static_rule_label = em.classic_heuristic(evidence_text)
        route, route_reason = em.recommend_review_route(
            hybrid_label=hybrid_label,
            hybrid_conf=hybrid_conf,
            rag_label=str(rag_result["label"]),
            rag_margin=float(rag_result["margin"]),
            static_rule_label=static_rule_label,
            profile=profile,
            cfg=self.model_cfg,
        )

        neighbors = self._serialize_neighbors(list(rag_result["neighbors"]))
        if llm_rag_top_k_override is None:
            llm_rag_top_k = max(1, int(self.cfg.llm_rag_top_k))
        else:
            llm_rag_top_k = max(0, int(llm_rag_top_k_override))
        prompt_neighbors = neighbors[:llm_rag_top_k] if include_rag_context_for_llm else []
        accessions = em._extract_accessions_for_prompt(evidence_text)
        accession_list = sorted(set(m.group(0).upper() for m in em.ACC_ANY.finditer(evidence_text)))
        gse_ids = [acc for acc in accession_list if acc.upper().startswith("GSE")]
        gse_urls = gse_urls_from_ids(gse_ids)
        standard_rag_context = em.build_standard_rag_context(
            accessions=accessions,
            structured_evidence=structured_summary[:1600],
            evidence=evidence_text[:1800],
            rag_neighbors=prompt_neighbors,
        )

        result: Dict[str, Any] = {
            "paper_id": paper_id or "",
            "title": title or "",
            "predictions": {
                "static_rules": static_rule_label,
                "linear_model": linear_label,
                "linear_model_primary_prob": float(linear_probs["prob_Primary"]),
                "linear_model_reuse_prob": float(linear_probs["prob_Reuse"]),
                "linear_model_plus_rag": hybrid_label,
                "linear_model_plus_rag_primary_prob": float(hybrid_probs["prob_Primary"]),
                "linear_model_plus_rag_reuse_prob": float(hybrid_probs["prob_Reuse"]),
                "linear_model_plus_rag_conf": hybrid_conf,
                "rag_vote": str(rag_result["label"]),
                "rag_vote_margin": float(rag_result["margin"]),
                "rag_vote_primary_score": float(rag_result["label_scores"].get("Primary", 0.0)),
                "rag_vote_reuse_score": float(rag_result["label_scores"].get("Reuse", 0.0)),
            },
            "recommended_route": route,
            "recommended_route_reason": route_reason,
            "evidence": {
                "accessions": accessions,
                "accession_list": accession_list,
                "gse_ids": gse_ids,
                "gse_urls": gse_urls,
                "evidence_text": evidence_text,
                "structured_evidence_summary": structured_summary,
                "structured_evidence_items": structured_json,
            },
            "rag": {
                "neighbors": neighbors,
                "standard_rag_context": standard_rag_context,
                "llm_rag_context_enabled": bool(include_rag_context_for_llm),
                "llm_rag_top_k": int(len(prompt_neighbors)),
                "rag_neighbors_available": int(len(neighbors)),
            },
            "final": {
                "label": hybrid_label,
                "source": "linear_model_plus_rag",
            },
            "decision_audit": {
                "baseline_label": hybrid_label,
                "baseline_source": "linear_model_plus_rag",
                "baseline_confidence": hybrid_conf,
                "llm_requested": bool(force_llm or use_llm),
                "llm_called": False,
                "llm_valid": False,
                "llm_strategy": llm_strategy,
                "llm_proposed_label": "",
                "llm_confidence": None,
                "llm_used_for_final": False,
                "changed_from_hybrid": False,
                "override_applied": False,
                "override_status": "not_requested",
                "llm_override_lock_applied": False,
                "llm_override_lock_threshold": float(self.cfg.llm_override_lock_threshold),
                "llm_advisory_only": False,
                "binary_label_reason": "",
            },
        }

        should_call_llm = force_llm or (use_llm and route == "llm_review")
        baseline_locked = (
            route == "auto_accept"
            and hybrid_conf >= float(self.cfg.llm_override_lock_threshold)
        )
        if baseline_locked:
            result["decision_audit"]["llm_override_lock_applied"] = True

        def keep_locked_baseline(parsed_label: str) -> None:
            result["final"] = {"label": hybrid_label, "source": "linear_model_plus_rag"}
            result["decision_audit"]["llm_used_for_final"] = False
            result["decision_audit"]["override_applied"] = False
            result["decision_audit"]["llm_advisory_only"] = True
            if parsed_label != hybrid_label:
                result["decision_audit"]["override_status"] = "blocked_high_confidence_baseline"
            else:
                result["decision_audit"]["override_status"] = "kept_high_confidence_baseline"

        if result["decision_audit"]["llm_requested"] and not should_call_llm:
            result["decision_audit"]["override_status"] = "not_called"
        if should_call_llm:
            if llm_strategy == "verify_override":
                prompt = em.build_rag_llm_verifier_prompt(
                    accessions=accessions,
                    evidence=evidence_text[:1800],
                    structured_evidence=structured_summary[:1600],
                    rag_neighbors=prompt_neighbors,
                    phrase_hints=self.phrase_hints,
                    base_label=str(rag_result["label"]),
                    rag_margin=float(rag_result["margin"]),
                    linear_label=hybrid_label,
                    static_rule_label=static_rule_label,
                )
            elif llm_strategy == "sentence_judge":
                prompt = em.build_rag_llm_sentence_judge_prompt(
                    accessions=accessions,
                    evidence=evidence_text[:1800],
                    structured_evidence=structured_summary[:1600],
                    rag_neighbors=prompt_neighbors,
                    phrase_hints=self.phrase_hints,
                )
            else:
                prompt = em.build_rag_llm_prompt(
                    accessions=accessions,
                    evidence=evidence_text[:1800],
                    structured_evidence=structured_summary[:1600],
                    rag_neighbors=prompt_neighbors,
                    phrase_hints=self.phrase_hints,
                )
            try:
                out = em.call_ollama_for_eval(prompt, self.model_cfg)
                llm_valid = str(out.get("rationale", "")).strip() != "No strict JSON found"
                parsed_label = str(out.get("label", hybrid_label))
                parsed_conf = float(out.get("confidence", 0.5))
                result["llm"] = {
                    "called": True,
                    "valid": bool(llm_valid),
                    "strategy": llm_strategy,
                    "output": out,
                }
                result["decision_audit"]["llm_called"] = True
                result["decision_audit"]["llm_valid"] = bool(llm_valid)
                result["decision_audit"]["llm_proposed_label"] = parsed_label
                result["decision_audit"]["llm_confidence"] = parsed_conf
                if llm_valid:
                    # High-confidence baseline rows are locked: the LLM may still be called,
                    # but its output is advisory and cannot overwrite the final label.
                    if baseline_locked:
                        keep_locked_baseline(parsed_label)
                        # Still preserve useful LLM-extracted sentence/evidence as advisory metadata.
                        llm_main_sentence = str(out.get("main_decision_sentence", "")).strip()
                        llm_main_role = normalize_main_decision_role(str(out.get("main_decision_role", "")))
                        if llm_main_sentence:
                            result["evidence"]["llm_main_decision_sentence_advisory"] = llm_main_sentence
                        if llm_main_role:
                            result["evidence"]["llm_main_decision_role_advisory"] = llm_main_role
                        if str(out.get("primary_evidence", "")).strip():
                            result["evidence"]["llm_primary_evidence"] = str(out.get("primary_evidence", "")).strip()
                        if str(out.get("reuse_evidence", "")).strip():
                            result["evidence"]["llm_reuse_evidence"] = str(out.get("reuse_evidence", "")).strip()
                    elif llm_strategy == "verify_override":
                        decision = str(out.get("decision", "")).strip().lower()
                        if decision == "override" or (parsed_label != str(rag_result["label"]) and parsed_conf >= 0.6):
                            result["final"] = {"label": parsed_label, "source": "rag_llm_verify_override"}
                            result["decision_audit"]["override_status"] = "used_changed_label" if parsed_label != hybrid_label else "used_no_change"
                        else:
                            result["final"] = {"label": hybrid_label, "source": "linear_model_plus_rag"}
                            result["decision_audit"]["override_status"] = "kept_baseline"
                    elif llm_strategy == "sentence_judge":
                        result["final"] = {"label": parsed_label, "source": "rag_llm_sentence_judge"}
                        result["decision_audit"]["override_status"] = "used_changed_label" if parsed_label != hybrid_label else "used_no_change"
                        llm_main_sentence = str(out.get("main_decision_sentence", "")).strip()
                        llm_main_role = normalize_main_decision_role(str(out.get("main_decision_role", "")))
                        if llm_main_sentence:
                            result["evidence"]["main_decision_sentence"] = llm_main_sentence
                        if llm_main_role:
                            result["evidence"]["main_decision_role"] = llm_main_role
                        if str(out.get("primary_evidence", "")).strip():
                            result["evidence"]["llm_primary_evidence"] = str(out.get("primary_evidence", "")).strip()
                        if str(out.get("reuse_evidence", "")).strip():
                            result["evidence"]["llm_reuse_evidence"] = str(out.get("reuse_evidence", "")).strip()
                    else:
                        result["final"] = {"label": parsed_label, "source": "rag_llm_classify"}
                        result["decision_audit"]["override_status"] = "used_changed_label" if parsed_label != hybrid_label else "used_no_change"
                        llm_main_sentence = str(out.get("main_decision_sentence", "")).strip()
                        if llm_main_sentence:
                            result["evidence"]["main_decision_sentence"] = llm_main_sentence
                        if str(out.get("primary_evidence", "")).strip():
                            result["evidence"]["llm_primary_evidence"] = str(out.get("primary_evidence", "")).strip()
                        if str(out.get("reuse_evidence", "")).strip():
                            result["evidence"]["llm_reuse_evidence"] = str(out.get("reuse_evidence", "")).strip()
                else:
                    result["decision_audit"]["override_status"] = "invalid_output"
            except Exception as exc:
                result["llm"] = {
                    "called": True,
                    "valid": False,
                    "strategy": llm_strategy,
                    "error": str(exc),
                }
                result["decision_audit"]["llm_called"] = True
                result["decision_audit"]["llm_valid"] = False
                result["decision_audit"]["override_status"] = "error"
        else:
            result["llm"] = {
                "called": False,
                "valid": False,
                "strategy": llm_strategy,
            }

        if not str(result["evidence"].get("main_decision_sentence", "")).strip():
            main_sentence, main_role = extract_main_decision_sentence(
                structured_items=structured_json,
                evidence_text=evidence_text,
                target_label=str(result.get("final", {}).get("label", hybrid_label)),
            )
            result["evidence"]["main_decision_sentence"] = main_sentence
            result["evidence"]["main_decision_role"] = main_role

        binary_final_label, binary_reason = coerce_binary_primary_precedence(
            label=str(result.get("final", {}).get("label", hybrid_label)),
            structured_items=structured_json,
            primary_evidence=str(result.get("evidence", {}).get("llm_primary_evidence", "")),
            reuse_evidence=str(result.get("evidence", {}).get("llm_reuse_evidence", "")),
        )
        result["final"]["label"] = binary_final_label
        result["decision_audit"]["binary_label_reason"] = binary_reason

        result["predictions"]["static_rules_raw"] = result["predictions"]["static_rules"]
        result["predictions"]["rag_vote_raw"] = result["predictions"]["rag_vote"]
        result["predictions"]["static_rules"] = coerce_binary_primary_precedence(
            label=str(result["predictions"]["static_rules"]),
            structured_items=structured_json,
        )[0]
        result["predictions"]["rag_vote"] = coerce_binary_primary_precedence(
            label=str(result["predictions"]["rag_vote"]),
            structured_items=structured_json,
        )[0]

        main_gse_ids = choose_main_decision_gse(
            main_decision_sentence=str(result["evidence"].get("main_decision_sentence", "")),
            main_decision_role=str(result["evidence"].get("main_decision_role", "")),
            final_label=binary_final_label,
            structured_items=structured_json,
            all_gse_ids=gse_ids,
        )
        result["evidence"]["main_decision_gse_ids"] = main_gse_ids
        result["evidence"]["main_decision_gse_urls"] = gse_urls_from_ids(main_gse_ids)

        result["decision_audit"]["llm_used_for_final"] = str(result.get("final", {}).get("source", "")).startswith("rag_llm")
        result["decision_audit"]["changed_from_hybrid"] = str(result.get("final", {}).get("label", "")) != coerce_binary_primary_precedence(
            label=hybrid_label,
            structured_items=structured_json,
        )[0]
        result["decision_audit"]["override_applied"] = bool(
            result["decision_audit"]["llm_used_for_final"] and result["decision_audit"]["changed_from_hybrid"]
        )

        return result

    def run_selected_models(
        self,
        *,
        title: str = "",
        text: str = "",
        paper_id: Optional[str] = None,
        identifier: str = "",
        modes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        requested_modes = list(
            modes or DEFAULT_RUNTIME_MODES
        )
        invalid = [mode for mode in requested_modes if mode not in SUPPORTED_RUNTIME_MODES]
        if invalid:
            raise ValueError(f"Unsupported model mode(s): {', '.join(invalid)}")
        if not requested_modes:
            raise ValueError("At least one model mode must be selected.")

        results: Dict[str, Any] = {}

        if not text.strip() and not identifier.strip():
            raise ValueError("run_selected_models requires either text or identifier.")
        for mode in requested_modes:
            results[mode] = self._classify_selected_mode(
                mode=mode,
                title=title,
                text=text,
                paper_id=paper_id,
                identifier=identifier,
            )

        preferred_mode = requested_modes[0]
        preferred = dict(results.get(preferred_mode) or {})
        preferred["selected_modes"] = requested_modes
        preferred["comparison_results"] = results
        preferred["comparison_labels"] = {
            mode: ((results.get(mode) or {}).get("final") or {}).get("label", "")
            for mode in requested_modes
        }
        return preferred

    def run_review_pipeline(
        self,
        *,
        title: str = "",
        text: str = "",
        paper_id: Optional[str] = None,
        identifier: str = "",
        base_mode: str = "hybrid_baseline",
        reviewer_mode: str = "none",
        include_rag_context_for_llm: bool = True,
        llm_rag_top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one baseline model, then optionally run an LLM reviewer layer.

        UI meaning:
        - base_mode chooses the main evidence-based baseline.
        - reviewer_mode controls whether an LLM is called after the baseline.
        - high-confidence override locking is still enforced inside classify().
        """
        if base_mode not in BASELINE_RUNTIME_MODES:
            raise ValueError(f"Unsupported base_mode: {base_mode}")
        if reviewer_mode not in ("none",) + tuple(LLM_RUNTIME_MODE_SPECS.keys()):
            raise ValueError(f"Unsupported reviewer_mode: {reviewer_mode}")
        if not text.strip() and not identifier.strip():
            raise ValueError("run_review_pipeline requires either text or identifier.")

        base_result = self._classify_selected_mode(
            mode=base_mode,
            title=title,
            text=text,
            paper_id=paper_id,
            identifier=identifier,
            include_rag_context_for_llm=include_rag_context_for_llm,
            llm_rag_top_k_override=llm_rag_top_k_override,
        )

        if reviewer_mode == "none":
            result = dict(base_result)
            result["displayed_mode"] = base_mode
            result["selected_modes"] = [base_mode]
            result["pipeline"] = {
                "base_mode": base_mode,
                "reviewer_mode": "none",
                "base_result": base_result,
                "reviewer_result": None,
                "policy": "baseline_only",
            }
            result["comparison_results"] = {base_mode: base_result}
            return result

        reviewer_result = self._classify_selected_mode(
            mode=reviewer_mode,
            title=title,
            text=text,
            paper_id=paper_id,
            identifier=identifier,
            include_rag_context_for_llm=include_rag_context_for_llm,
            llm_rag_top_k_override=llm_rag_top_k_override,
        )

        composed = dict(reviewer_result)
        base_final = dict((base_result.get("final") or {}))
        reviewer_final = dict((reviewer_result.get("final") or {}))
        audit = dict((reviewer_result.get("decision_audit") or {}))

        llm_used = bool(audit.get("llm_used_for_final"))
        advisory_only = bool(audit.get("llm_advisory_only"))
        if llm_used and not advisory_only:
            composed["final"] = reviewer_final
            pipeline_policy = "reviewer_set_final"
        else:
            composed["final"] = base_final
            pipeline_policy = "base_kept_reviewer_advisory"

        audit["pipeline_base_mode"] = base_mode
        audit["pipeline_reviewer_mode"] = reviewer_mode
        audit["llm_rag_context_enabled"] = bool(include_rag_context_for_llm)
        audit["llm_rag_top_k"] = int((reviewer_result.get("rag") or {}).get("llm_rag_top_k", 0))
        audit["pipeline_policy"] = pipeline_policy
        audit["pipeline_base_label"] = str(base_final.get("label", ""))
        audit["pipeline_base_source"] = str(base_final.get("source", ""))
        composed["decision_audit"] = audit
        composed["displayed_mode"] = "pipeline"
        composed["selected_modes"] = [base_mode, reviewer_mode]
        composed["pipeline"] = {
            "base_mode": base_mode,
            "reviewer_mode": reviewer_mode,
            "base_result": base_result,
            "reviewer_result": reviewer_result,
            "include_rag_context_for_llm": bool(include_rag_context_for_llm),
            "llm_rag_top_k": int((reviewer_result.get("rag") or {}).get("llm_rag_top_k", 0)),
            "policy": pipeline_policy,
        }
        # Important: do not put `composed` itself inside `comparison_results`.
        # That creates a self-referential object that FastAPI cannot JSON-encode
        # and causes HTTP 500 responses in the UI. Build a shallow, acyclic
        # inspection card for the composed pipeline final instead.
        pipeline_final_card = {
            "paper_id": composed.get("paper_id", ""),
            "title": composed.get("title", ""),
            "final": composed.get("final", {}),
            "predictions": composed.get("predictions", {}),
            "recommended_route": composed.get("recommended_route", ""),
            "recommended_route_reason": composed.get("recommended_route_reason", ""),
            "evidence": composed.get("evidence", {}),
            "rag": composed.get("rag", {}),
            "llm": composed.get("llm", {}),
            "decision_audit": composed.get("decision_audit", {}),
            "displayed_mode": "pipeline_final",
        }
        composed["comparison_results"] = {
            base_mode: base_result,
            reviewer_mode: reviewer_result,
            "pipeline_final": pipeline_final_card,
        }
        composed["comparison_labels"] = {
            base_mode: str(base_final.get("label", "")),
            reviewer_mode: str(reviewer_final.get("label", "")),
            "pipeline_final": str((composed.get("final") or {}).get("label", "")),
        }
        return composed

    def classify_identifier(
        self,
        identifier: str,
        use_llm: bool = False,
        force_llm: bool = False,
        llm_strategy: str = "classify",
        include_rag_context_for_llm: bool = True,
        llm_rag_top_k_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        article = self.lookup_article(identifier)
        if article is None:
            return {
                "identifier": identifier,
                "found": False,
                "message": "Identifier not found in the local article index. DOI lookup is only supported when DOI exists in local indexed metadata.",
            }
        result = self.classify(
            title=str(article.get("title", "")),
            text=str(article.get("text", "")),
            paper_id=str(article.get("paper_id", "")),
            use_llm=use_llm,
            force_llm=force_llm,
            llm_strategy=llm_strategy,
            include_rag_context_for_llm=include_rag_context_for_llm,
            llm_rag_top_k_override=llm_rag_top_k_override,
        )
        result["found"] = True
        result["lookup"] = {
            "identifier": identifier,
            "paper_id": str(article.get("paper_id", "")),
            "article_url": str(article.get("article_url", "")),
            "source_gse_ids": list(article.get("gse_ids", [])),
            "source_gse_urls": gse_urls_from_ids(list(article.get("gse_ids", []))),
            "doi": str(article.get("doi", "")),
            "pmcid": str(article.get("pmcid", "")),
            "source": str(article.get("source", "")),
        }
        return result

    def classify_batch_records(
        self,
        records: List[Dict[str, Any]],
        use_llm: bool = False,
        force_llm: bool = False,
        llm_strategy: str = "classify",
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(records)
        for idx, record in enumerate(records, start=1):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Batch job cancelled by user.")
            text = str(record.get("text", "") or record.get("full_text", "") or record.get("article_text", "")).strip()
            title = str(record.get("title", "")).strip()
            paper_id = str(record.get("paper_id", "")).strip() or None
            identifier = str(record.get("identifier", "") or record.get("pmcid", "") or record.get("doi", "")).strip()
            if text:
                result = self.classify(
                    title=title,
                    text=text,
                    paper_id=paper_id,
                    use_llm=use_llm,
                    force_llm=force_llm,
                    llm_strategy=llm_strategy,
                )
            elif identifier:
                result = self.classify_identifier(
                    identifier=identifier,
                    use_llm=use_llm,
                    force_llm=force_llm,
                    llm_strategy=llm_strategy,
                )
            elif paper_id:
                result = self.classify_identifier(
                    identifier=paper_id,
                    use_llm=use_llm,
                    force_llm=force_llm,
                    llm_strategy=llm_strategy,
                )
            else:
                result = {
                    "found": False,
                    "message": "Row must include either text, paper_id, pmcid, doi, or identifier.",
                    "input_record": record,
                }
            result["input_record"] = record
            results.append(result)
            if progress_callback is not None:
                progress_callback(
                    idx,
                    total,
                    {
                        "current_row": idx,
                        "total_rows": total,
                        "current_mode": llm_strategy if use_llm or force_llm else "baseline",
                        "current_identifier": identifier or paper_id or title or f"row_{idx}",
                    },
                )
        return results

    def classify_batch_selected_models(
        self,
        records: List[Dict[str, Any]],
        modes: List[str],
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total_rows = len(records)
        total_units = max(1, len(records) * max(1, len(modes)))
        completed_units = 0
        for idx, record in enumerate(records, start=1):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Batch job cancelled by user.")
            text = str(record.get("text", "") or record.get("full_text", "") or record.get("article_text", "")).strip()
            title = str(record.get("title", "")).strip()
            paper_id = str(record.get("paper_id", "")).strip() or None
            identifier = str(record.get("identifier", "") or record.get("pmcid", "") or record.get("doi", "")).strip()
            try:
                comparison_results: Dict[str, Any] = {}
                if progress_callback is not None:
                    progress_callback(
                        completed_units,
                        total_units,
                        {
                            "current_row": idx,
                            "total_rows": total_rows,
                            "current_mode": "starting_row",
                            "current_identifier": identifier or paper_id or title or f"row_{idx}",
                        },
                    )
                for mode in modes:
                    comparison_results[mode] = self._classify_selected_mode(
                        mode=mode,
                        title=title,
                        text=text,
                        paper_id=paper_id,
                        identifier=identifier,
                    )
                    completed_units += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_units,
                            total_units,
                            {
                                "current_row": idx,
                                "total_rows": total_rows,
                                "current_mode": mode,
                                "current_identifier": identifier or paper_id or title or f"row_{idx}",
                            },
                        )
                preferred_mode = modes[0]
                result = dict(comparison_results.get(preferred_mode) or {})
                result["selected_modes"] = list(modes)
                result["comparison_results"] = comparison_results
                result["comparison_labels"] = {
                    mode: ((comparison_results.get(mode) or {}).get("final") or {}).get("label", "")
                    for mode in modes
                }
            except Exception as exc:
                result = {
                    "found": False,
                    "message": str(exc),
                    "input_record": record,
                    "selected_modes": list(modes),
                    "comparison_results": {},
                    "comparison_labels": {},
                }
            result["input_record"] = record
            results.append(result)
        return results

    def save_feedback(
        self,
        *,
        paper_id: Optional[str],
        identifier: Optional[str],
        title: str,
        text: str,
        predicted_label: str,
        corrected_label: str,
        reviewer: str = "",
        note: str = "",
    ) -> Dict[str, Any]:
        text_value = text
        title_value = title
        lookup_key = paper_id or identifier or ""
        if not text_value.strip() and lookup_key:
            article = self.lookup_article(lookup_key)
            if article is not None:
                text_value = str(article.get("text", ""))
                if not title_value.strip():
                    title_value = str(article.get("title", ""))
        if not text_value.strip():
            raise ValueError("Feedback requires either raw text or a locally resolvable paper identifier.")
        inference = self.classify(
            title=title_value,
            text=text_value,
            paper_id=paper_id,
            use_llm=False,
            force_llm=False,
            llm_strategy="classify",
        )
        row = {
            "paper_id": paper_id or "",
            "identifier": identifier or "",
            "title": title_value,
            "predicted_label": predicted_label,
            "corrected_label": corrected_label,
            "feedback_decision": "confirmed_correct" if str(predicted_label).strip() == str(corrected_label).strip() else "corrected_label",
            "reviewer": reviewer,
            "note": note,
            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "recommended_route": inference.get("recommended_route", ""),
            "recommended_route_reason": inference.get("recommended_route_reason", ""),
            "pred_linear_model_plus_rag": inference.get("predictions", {}).get("linear_model_plus_rag", ""),
            "pred_linear_model_plus_rag_conf": inference.get("predictions", {}).get("linear_model_plus_rag_conf", ""),
            "pred_rag_vote": inference.get("predictions", {}).get("rag_vote", ""),
            "gse_ids": ";".join(inference.get("evidence", {}).get("gse_ids", [])),
            "gse_urls": ";".join(inference.get("evidence", {}).get("gse_urls", [])),
            "main_decision_gse_ids": ";".join(inference.get("evidence", {}).get("main_decision_gse_ids", [])),
            "main_decision_gse_urls": ";".join(inference.get("evidence", {}).get("main_decision_gse_urls", [])),
            "accession_list": ";".join(inference.get("evidence", {}).get("accession_list", [])),
            "main_decision_sentence": inference.get("evidence", {}).get("main_decision_sentence", ""),
            "main_decision_role": inference.get("evidence", {}).get("main_decision_role", ""),
            "structured_evidence_summary": inference.get("evidence", {}).get("structured_evidence_summary", ""),
            "evidence_text": inference.get("evidence", {}).get("evidence_text", "")[:4000],
        }
        self.feedback_store_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.feedback_store_path.exists()
        with self.feedback_store_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return {
            "saved": True,
            "feedback_store_path": str(self.feedback_store_path),
            "row": row,
        }

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
        use_llm: bool = False,
        force_llm: bool = False,
        llm_strategy: str = "classify",
    ) -> List[Dict[str, Any]]:
        return [
            self.classify(
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                paper_id=str(item.get("paper_id", "")) or None,
                use_llm=use_llm,
                force_llm=force_llm,
                llm_strategy=llm_strategy,
            )
            for item in items
        ]
