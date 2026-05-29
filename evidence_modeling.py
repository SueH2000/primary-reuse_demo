#!/usr/bin/env python3
"""Systematic evidence modeling for Primary vs Reuse classification.

This script provides a reproducible upgrade path:
1) Auto phrase mining from labeled data (top-k phrases per class)
2) Convert mined phrases into regex pattern templates
3) Train a sentence-level linear model on evidence text
4) Compare all methods on the same train/test split

Important data assumption:
- You can provide article content in JSONL and labels in CSV.
- They are matched by `paper_id` (recommended setup).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, train_test_split

ACC_ANY = re.compile(r"(?i)\b(GSE\d+|GSM\d+|SRP\d+|SRR\d+|E-\w{2,3}-\d+|PRJ[EDNA][A-Z0-9]+)\b")
GEO_WORDS = re.compile(r"(?i)\b(GEO|Gene Expression Omnibus|SRA|ArrayExpress|ENA|NCBI)\b")
PROV_WORDS = re.compile(
    r"(?i)\b(downloaded|retrieved|obtained|reanaly[sz]ed|re-analysed|publicly available|"
    r"deposited|submitted|accession|available at|data availability|data are available)\b"
)
PRIMARY_STRONG = re.compile(
    r"(?i)\b(we (collected|recruited|enrolled|sequenced|generated|performed rna-?seq|acquired)|"
    r"library preparation|sample collection|patients were recruited|ethics approval|informed consent|our cohort)\b"
)
REUSE_STRONG = re.compile(
    r"(?i)\b(downloaded|retrieved|obtained from|publicly available|reanaly[sz]ed|secondary analysis)\b"
)
DEPOSIT = re.compile(r"(?i)\b(deposited|submitted)\b")
WE_OUR = re.compile(r"(?i)\b(we|our)\b")
PRIMARY_CUE_WORDS = re.compile(
    r"(?i)\b(generated|sequenced|collected|recruited|enrolled|measured|profiled|performed rna-?seq|"
    r"library preparation|sample collection|our cohort|our samples?|patients were recruited)\b"
)
REUSE_CUE_WORDS = re.compile(
    r"(?i)\b(downloaded|retrieved|obtained from|acquired from|publicly available|re-?analy[sz]ed|"
    r"integrated analysis|geo dataset|geo datasets|from geo|from the gene expression omnibus|from the geo database)\b"
)
DEPOSIT_CUE_WORDS = re.compile(r"(?i)\b(deposited|submitted|accession number|geo accession|available at)\b")
DOWNLOAD_CUE_WORDS = re.compile(
    r"(?i)\b(downloaded|retrieved|obtained from|acquired from|publicly available|from geo|from the gene expression omnibus|from the geo database)\b"
)
REANALYSIS_CUE_WORDS = re.compile(
    r"(?i)\b(re-?analy[sz]ed|secondary analysis|integrated analysis|combined analysis|meta-analysis|public datasets?)\b"
)
STRUCTURED_ROLE_ORDER = [
    "Primary_generation",
    "Primary_deposition",
    "Reuse_download",
    "Reuse_reanalysis",
    "Mixed_conflict",
    "Other_provenance",
]


@dataclass
class ModelingConfig:
    labeled_csv_path: Path
    out_dir: Path
    jsonl_path: Optional[Path] = None
    test_size: float = 0.2
    random_state: int = 42
    top_k_phrases: int = 40
    ngram_min: int = 1
    ngram_max: int = 3
    min_df: int = 2
    max_features: int = 20000
    extraction_mode: str = "accession_windows"
    win_before: int = 350
    win_after: int = 900
    max_evidence_chars: int = 2200
    cv_folds: int = 5
    rag_top_k: int = 8
    rag_candidate_pool: int = 12
    rag_per_label_cap: int = 4
    rag_min_similarity: float = 0.01
    rag_vote_margin: float = 0.0
    llm_eval_mode: str = "off"
    llm_strategy: str = "verify_override"
    llm_hard_margin: float = 0.05
    llm_eval_max_rows: int = 0
    llm_eval_in_cv: bool = False
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3"
    llm_temperature: float = 0.0
    llm_num_predict: int = 256
    llm_rag_top_k: int = 3
    llm_timeout_connect: int = 10
    llm_timeout_read: int = 300
    route_auto_accept_threshold: float = 0.85
    route_llm_review_threshold: float = 0.60
    split_path: Optional[Path] = None
    save_split_path: Optional[Path] = None


def normalize_label(x: Any) -> str:
    if pd.isna(x):
        return "Unclear"
    s = str(x).strip().lower()
    if s in {"primary", "p", "generated", "own", "new"} or "primary" in s:
        return "Primary"
    if s in {"reuse", "re-used", "reused", "secondary", "public", "old"} or "reuse" in s:
        return "Reuse"
    return "Unclear"


def read_csv_flex(path: Path) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_exc


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] JSON decode failed at line {i}: {exc}")


def extract_text_fields(rec: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ["title", "abstract", "full_text", "body", "text"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def evidence_windows(
    text: str,
    win_before: int = 350,
    win_after: int = 900,
    max_chars: int = 2200,
    max_hits: int = 40,
) -> str:
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return ""

    hits: List[str] = []
    for m in ACC_ANY.finditer(flat):
        s = max(0, m.start() - win_before)
        e = min(len(flat), m.end() + win_after)
        chunk = flat[s:e]
        if GEO_WORDS.search(chunk) or PROV_WORDS.search(chunk):
            hits.append(chunk)
        if len(hits) >= max_hits:
            break

    if not hits:
        for m in PROV_WORDS.finditer(flat):
            s = max(0, m.start() - win_before)
            e = min(len(flat), m.end() + win_after)
            hits.append(flat[s:e])
            if len(hits) >= max_hits:
                break

    dedup, seen = [], set()
    for h in hits:
        key = h[:140].lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)

    evidence = re.sub(r"\s+", " ", " ... ".join(dedup)).strip()
    return evidence[:max_chars]


SECTION_HINTS = re.compile(
    r"(?i)\b(data availability|availability of data|materials? and methods?|methods?|patients? and methods?|"
    r"rna-?seq|microarray|sequencing|geo accession|accession number|dataset|datasets|deposited|submitted)\b"
)
SECTION_SPLIT = re.compile(r"(?:(?<=\.)\s+|\n+)")


def section_aware_evidence(
    text: str,
    max_chars: int = 2200,
    max_windows: int = 8,
) -> str:
    """Prefer high-value provenance spans instead of only fixed accession windows.

    Why this helps:
    - provenance clues often live in Methods/Data Availability, not just near one accession
    - some papers mention both generated and reused data, so we want several focused spans
    - keeping the output short preserves signal for retrieval and downstream classifiers
    """
    if not text:
        return ""

    parts = [p.strip() for p in SECTION_SPLIT.split(text) if p and p.strip()]
    scored: List[Tuple[int, int, str]] = []

    for idx, part in enumerate(parts):
        p = re.sub(r"\s+", " ", part).strip()
        if len(p) < 40:
            continue

        score = 0
        if ACC_ANY.search(p):
            score += 4
        if PROV_WORDS.search(p):
            score += 3
        if SECTION_HINTS.search(p):
            score += 2
        if PRIMARY_STRONG.search(p):
            score += 2
        if REUSE_STRONG.search(p):
            score += 2
        if GEO_WORDS.search(p):
            score += 2
        if score <= 0:
            continue

        left = max(0, idx - 1)
        right = min(len(parts), idx + 2)
        window = " ".join(re.sub(r"\s+", " ", x).strip() for x in parts[left:right])
        scored.append((score, idx, window))

    scored.sort(key=lambda x: (-x[0], x[1]))

    chosen: List[str] = []
    seen = set()
    total = 0
    for _, _, window in scored:
        key = window[:180].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(window)
        total += len(window)
        if len(chosen) >= max_windows or total >= max_chars:
            break

    if not chosen:
        return evidence_windows(text, max_chars=max_chars)

    evidence = re.sub(r"\s+", " ", " ... ".join(chosen)).strip()
    return evidence[:max_chars]


def build_evidence_text(
    text: str,
    extraction_mode: str = "accession_windows",
    max_chars: int = 2200,
    win_before: int = 350,
    win_after: int = 900,
) -> str:
    """Dispatch between evidence extraction strategies."""
    mode = (extraction_mode or "accession_windows").strip().lower()
    if mode == "accession_windows":
        return evidence_windows(text, win_before=win_before, win_after=win_after, max_chars=max_chars)
    if mode == "section_aware":
        return section_aware_evidence(text, max_chars=max_chars)
    raise ValueError(f"Unsupported extraction_mode: {extraction_mode}")


def classic_heuristic(text: str) -> str:
    if REUSE_STRONG.search(text):
        return "Reuse"
    if PRIMARY_STRONG.search(text):
        return "Primary"
    if DEPOSIT.search(text) and WE_OUR.search(text):
        return "Primary"
    return "Unclear"


def provenance_chunks(text: str, max_chunks: int = 8) -> List[str]:
    """Return short provenance-dense spans for retrieval and prompting."""
    if not text:
        return []

    parts = [p.strip() for p in SECTION_SPLIT.split(text) if p and p.strip()]
    scored: List[Tuple[int, int, str]] = []
    for idx, part in enumerate(parts):
        p = re.sub(r"\s+", " ", part).strip()
        if len(p) < 20:
            continue
        score = 0
        if ACC_ANY.search(p):
            score += 4
        if GEO_WORDS.search(p):
            score += 3
        if REUSE_CUE_WORDS.search(p):
            score += 3
        if PRIMARY_CUE_WORDS.search(p):
            score += 3
        if DEPOSIT_CUE_WORDS.search(p):
            score += 2
        if PROV_WORDS.search(p):
            score += 2
        if score <= 0:
            continue
        left = max(0, idx - 1)
        right = min(len(parts), idx + 2)
        window = " ".join(re.sub(r"\s+", " ", x).strip() for x in parts[left:right])
        scored.append((score, idx, window))

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: List[str] = []
    seen = set()
    for _, _, chunk in scored:
        key = chunk[:180].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(chunk)
        if len(chosen) >= max_chunks:
            break
    return chosen


def classify_evidence_role(chunk: str) -> Tuple[str, str]:
    """Map one provenance chunk to a structured role and coarse label."""
    flat = re.sub(r"\s+", " ", chunk or "").strip()
    has_primary = bool(PRIMARY_CUE_WORDS.search(flat))
    has_deposit = bool(DEPOSIT_CUE_WORDS.search(flat))
    has_download = bool(DOWNLOAD_CUE_WORDS.search(flat))
    has_reanalysis = bool(REANALYSIS_CUE_WORDS.search(flat))
    has_geo = bool(GEO_WORDS.search(flat) or ACC_ANY.search(flat))
    has_we_our = bool(WE_OUR.search(flat))
    looks_like_results_submission = bool(
        re.search(r"(?i)\b(all|raw|processed|microarray|rna-?seq|sequencing|expression)\s+(data|results|files)\b", flat)
    )

    if (has_primary or (has_deposit and has_we_our)) and (has_download or has_reanalysis):
        return "Mixed_conflict", "Unclear"
    if has_primary:
        return "Primary_generation", "Primary"
    if has_deposit and (has_we_our or looks_like_results_submission) and not has_download:
        return "Primary_deposition", "Primary"
    if has_download and has_geo:
        return "Reuse_download", "Reuse"
    if has_reanalysis or (has_download and not has_we_our):
        return "Reuse_reanalysis", "Reuse"
    return "Other_provenance", "Unclear"


def structured_evidence_items(text: str, max_items: int = 8) -> List[Dict[str, Any]]:
    """Convert extracted evidence into explicit role-tagged spans."""
    items: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(provenance_chunks(text, max_chunks=max_items), start=1):
        role, coarse_label = classify_evidence_role(chunk)
        accessions = sorted(set(m.group(0).upper() for m in ACC_ANY.finditer(chunk)))
        items.append(
            {
                "rank": idx,
                "role": role,
                "coarse_label": coarse_label,
                "accessions": accessions[:6],
                "text": chunk[:320],
            }
        )
    return items


def structured_evidence_summary(items: List[Dict[str, Any]], max_items: int = 5) -> str:
    if not items:
        return ""
    rows = []
    for item in items[:max_items]:
        accessions = ",".join(item.get("accessions", [])[:3]) or "none"
        rows.append(f"[{item['role']}] acc={accessions} text={item['text']}")
    return " || ".join(rows)


def structured_role_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {role: 0 for role in STRUCTURED_ROLE_ORDER}
    for item in items:
        role = str(item.get("role", "Other_provenance"))
        counts[role] = counts.get(role, 0) + 1
    return counts


def enrich_with_structured_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Attach explicit role-tagged evidence columns derived from evidence_text."""
    work = df.copy()
    items_series = work["evidence_text"].fillna("").astype(str).apply(structured_evidence_items)
    work["structured_evidence_items"] = items_series
    work["structured_evidence_summary"] = items_series.apply(structured_evidence_summary)
    work["structured_evidence_json"] = items_series.apply(lambda items: json.dumps(items, ensure_ascii=False))
    return work


def provenance_profile(text: str) -> Dict[str, Any]:
    """Summarize provenance cues so retrieval can prefer label-aligned neighbors."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    accessions = sorted(set(m.group(0).upper() for m in ACC_ANY.finditer(flat)))
    items = structured_evidence_items(flat)
    role_counts = structured_role_counts(items)
    return {
        "primary_cues": len(PRIMARY_CUE_WORDS.findall(flat)),
        "reuse_cues": len(REUSE_CUE_WORDS.findall(flat)),
        "deposit_cues": len(DEPOSIT_CUE_WORDS.findall(flat)),
        "download_cues": len(DOWNLOAD_CUE_WORDS.findall(flat)),
        "geo_mentions": len(GEO_WORDS.findall(flat)),
        "accession_count": len(accessions),
        "accessions": accessions[:10],
        "has_we_our": int(bool(WE_OUR.search(flat))),
        "role_counts": role_counts,
        "structured_summary": structured_evidence_summary(items),
    }


def build_provenance_retrieval_text(title: str, evidence_text: str) -> str:
    """Compress a paper into provenance-focused retrieval text."""
    chunks = provenance_chunks(evidence_text)
    focus = " ".join(chunks) if chunks else re.sub(r"\s+", " ", evidence_text or "").strip()
    profile = provenance_profile(evidence_text)
    cue_tokens: List[str] = []
    cue_tokens.extend(["cue_primary"] * min(int(profile["primary_cues"]), 3))
    cue_tokens.extend(["cue_reuse"] * min(int(profile["reuse_cues"]), 3))
    cue_tokens.extend(["cue_deposit"] * min(int(profile["deposit_cues"]), 2))
    cue_tokens.extend(["cue_download"] * min(int(profile["download_cues"]), 2))
    cue_tokens.extend(["cue_geo"] * min(int(profile["geo_mentions"]), 2))
    if int(profile["has_we_our"]):
        cue_tokens.append("cue_we_our")
    for role_name, count in profile.get("role_counts", {}).items():
        cue_tokens.extend([f"role_{role_name.lower()}"] * min(int(count), 2))
    accessions = " ".join(profile["accessions"])
    structured = str(profile.get("structured_summary", "")).strip()
    title_tail = f" title:{title}" if title else ""
    return re.sub(r"\s+", " ", f"{focus} {structured} {' '.join(cue_tokens)} {accessions}{title_tail}").strip()


def provenance_alignment_score(query_profile: Dict[str, Any], candidate_profile: Dict[str, Any]) -> float:
    """Hybrid retrieval bonus/penalty based on provenance cue agreement."""
    score = 0.0
    if query_profile["reuse_cues"] and candidate_profile["reuse_cues"]:
        score += 0.12
    if query_profile["primary_cues"] and candidate_profile["primary_cues"]:
        score += 0.12
    if query_profile["deposit_cues"] and candidate_profile["deposit_cues"]:
        score += 0.05
    if query_profile["download_cues"] and candidate_profile["download_cues"]:
        score += 0.07
    if query_profile["geo_mentions"] and candidate_profile["geo_mentions"]:
        score += 0.04
    if query_profile["has_we_our"] and candidate_profile["has_we_our"]:
        score += 0.03

    accession_overlap = len(set(query_profile["accessions"]) & set(candidate_profile["accessions"]))
    score += 0.05 * min(accession_overlap, 2)

    query_roles = query_profile.get("role_counts", {})
    candidate_roles = candidate_profile.get("role_counts", {})
    for role_name in ["Reuse_download", "Reuse_reanalysis", "Primary_generation", "Primary_deposition"]:
        if query_roles.get(role_name, 0) and candidate_roles.get(role_name, 0):
            score += 0.06

    if query_profile["reuse_cues"] and candidate_profile["primary_cues"] and not candidate_profile["reuse_cues"]:
        score -= 0.08
    if query_profile["primary_cues"] and candidate_profile["reuse_cues"] and not candidate_profile["primary_cues"]:
        score -= 0.08
    if query_roles.get("Reuse_download", 0) and candidate_roles.get("Primary_generation", 0):
        score -= 0.06
    if query_roles.get("Primary_generation", 0) and candidate_roles.get("Reuse_download", 0):
        score -= 0.06
    return float(score)


def rag_feature_row(result: Dict[str, Any], profile: Dict[str, Any]) -> List[float]:
    neighbors = list(result.get("neighbors", []))
    role_counts = profile.get("role_counts", {})
    return [
        float(result.get("label_scores", {}).get("Primary", 0.0)),
        float(result.get("label_scores", {}).get("Reuse", 0.0)),
        float(result.get("margin", 0.0)),
        float(result.get("best_score", 0.0)),
        float(len(neighbors)),
        float(sum(1 for n in neighbors if str(n.get("label")) == "Primary")),
        float(sum(1 for n in neighbors if str(n.get("label")) == "Reuse")),
        float(profile.get("primary_cues", 0)),
        float(profile.get("reuse_cues", 0)),
        float(profile.get("deposit_cues", 0)),
        float(profile.get("download_cues", 0)),
        float(profile.get("geo_mentions", 0)),
        float(profile.get("accession_count", 0)),
        float(profile.get("has_we_our", 0)),
        float(role_counts.get("Primary_generation", 0)),
        float(role_counts.get("Primary_deposition", 0)),
        float(role_counts.get("Reuse_download", 0)),
        float(role_counts.get("Reuse_reanalysis", 0)),
        float(role_counts.get("Mixed_conflict", 0)),
    ]


def _safe_pattern_from_phrase(phrase: str) -> str:
    tokens = [re.escape(t) for t in phrase.lower().split() if t.strip()]
    if not tokens:
        return ""
    return r"\b" + r"\s+".join(tokens) + r"\b"


def mine_top_phrases(
    texts: pd.Series,
    labels: pd.Series,
    top_k: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_features: int,
) -> Dict[str, List[Dict[str, float]]]:
    vec = CountVectorizer(
        ngram_range=ngram_range,
        lowercase=True,
        stop_words="english",
        min_df=min_df,
        max_features=max_features,
    )
    X = vec.fit_transform(texts.fillna(""))
    vocab = np.array(vec.get_feature_names_out())

    results: Dict[str, List[Dict[str, float]]] = {}
    eps = 1.0
    label_values = labels.to_numpy()

    for cls in ["Primary", "Reuse"]:
        mask_cls = label_values == cls
        mask_other = label_values != cls
        c_cls = np.asarray(X[mask_cls].sum(axis=0)).ravel() + eps
        c_oth = np.asarray(X[mask_other].sum(axis=0)).ravel() + eps
        p_cls = c_cls / c_cls.sum()
        p_oth = c_oth / c_oth.sum()
        log_odds = np.log(p_cls / p_oth)

        idx = np.argsort(-log_odds)[:top_k]
        results[cls] = [{"phrase": str(vocab[i]), "score": float(log_odds[i])} for i in idx]

    return results


def build_template_rules(mined: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[re.Pattern[str]]]:
    rules: Dict[str, List[re.Pattern[str]]] = {"Primary": [], "Reuse": []}
    for cls in ["Primary", "Reuse"]:
        for item in mined.get(cls, []):
            patt = _safe_pattern_from_phrase(item["phrase"])
            if patt:
                rules[cls].append(re.compile(patt, re.IGNORECASE))
    return rules


def template_rule_predict(text: str, rules: Dict[str, List[re.Pattern[str]]]) -> str:
    t = text or ""
    p_score = sum(1 for r in rules["Primary"] if r.search(t))
    r_score = sum(1 for r in rules["Reuse"] if r.search(t))
    if p_score > r_score:
        return "Primary"
    if r_score > p_score:
        return "Reuse"
    return "Unclear"


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    all_acc = float(accuracy_score(y_true, y_pred))
    mask = y_true.isin(["Primary", "Reuse"]) & y_pred.isin(["Primary", "Reuse"])
    binary_acc = float(accuracy_score(y_true[mask], y_pred[mask])) if int(mask.sum()) > 0 else np.nan
    return {
        "accuracy_all": all_acc,
        "accuracy_binary_on_covered": binary_acc,
        "coverage_binary_pred": float(y_pred.isin(["Primary", "Reuse"]).mean()),
        "unclear_rate": float((y_pred == "Unclear").mean()),
        "n": int(len(y_true)),
    }


def _probability_frame(clf: LogisticRegression, X: Any) -> pd.DataFrame:
    """Return a class-probability DataFrame aligned to Primary/Reuse labels."""
    classes = [str(x) for x in clf.classes_]
    probs = clf.predict_proba(X)
    df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
    for cls in ["Primary", "Reuse"]:
        col = f"prob_{cls}"
        if col not in df.columns:
            df[col] = 0.0
    return df[["prob_Primary", "prob_Reuse"]]


def has_mixed_provenance_signal(profile: Dict[str, Any]) -> bool:
    """Detect rows where both primary-like and reuse-like role signals appear."""
    role_counts = profile.get("role_counts", {})
    has_primary_role = role_counts.get("Primary_generation", 0) > 0 or role_counts.get("Primary_deposition", 0) > 0
    has_reuse_role = role_counts.get("Reuse_download", 0) > 0 or role_counts.get("Reuse_reanalysis", 0) > 0
    return bool(role_counts.get("Mixed_conflict", 0) > 0 or (has_primary_role and has_reuse_role))


def recommend_review_route(
    hybrid_label: str,
    hybrid_conf: float,
    rag_label: str,
    rag_margin: float,
    static_rule_label: str,
    profile: Dict[str, Any],
    cfg: ModelingConfig,
) -> Tuple[str, str]:
    """Recommend production routing: auto_accept, llm_review, or human_review."""
    reasons: List[str] = []
    mixed_signal = has_mixed_provenance_signal(profile)
    static_disagree = static_rule_label in {"Primary", "Reuse"} and static_rule_label != hybrid_label
    rag_disagree = rag_label in {"Primary", "Reuse"} and rag_label != hybrid_label
    low_margin = rag_margin < cfg.llm_hard_margin

    if hybrid_conf < cfg.route_llm_review_threshold:
        reasons.append("low_hybrid_confidence")
        if mixed_signal:
            reasons.append("mixed_structured_roles")
        if static_disagree:
            reasons.append("static_rule_disagree")
        if rag_disagree:
            reasons.append("rag_disagree")
        return "human_review", "|".join(reasons)

    if mixed_signal:
        reasons.append("mixed_structured_roles")
    if static_disagree:
        reasons.append("static_rule_disagree")
    if rag_disagree:
        reasons.append("rag_disagree")
    if low_margin:
        reasons.append("low_rag_margin")

    if not reasons and hybrid_conf >= cfg.route_auto_accept_threshold:
        return "auto_accept", "high_hybrid_confidence"

    if not reasons and hybrid_conf >= cfg.route_llm_review_threshold:
        return "auto_accept", "medium_confidence_clean"

    return "llm_review", "|".join(reasons) if reasons else "medium_confidence"


def build_llm_evaluation_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much of rag_llm is real LLM output versus fallback.

    Why this matters:
    - `rag_llm` can look artificially strong when most rows quietly fall back to `rag_vote`
    - we therefore report both all-row performance and the subset where the LLM was actually called
    """
    required = {"label", "pred_rag_vote", "pred_rag_llm", "pred_rag_llm_called"}
    if not required.issubset(test_df.columns):
        return pd.DataFrame()

    llm_called = test_df["pred_rag_llm_called"].fillna(0).astype(int) == 1
    llm_valid = (
        test_df["pred_rag_llm_valid"].fillna(0).astype(int) == 1
        if "pred_rag_llm_valid" in test_df.columns
        else llm_called
    )
    scopes = [
        ("all_rows", pd.Series(True, index=test_df.index)),
        ("llm_called_only", llm_called),
        ("llm_valid_only", llm_valid),
    ]
    rows: List[Dict[str, Any]] = []

    for scope_name, mask in scopes:
        n_scope = int(mask.sum())
        if n_scope == 0:
            continue

        scope_df = test_df.loc[mask].copy()
        for model_name, pred_col in [("rag_vote", "pred_rag_vote"), ("rag_llm", "pred_rag_llm")]:
            metrics = evaluate_predictions(scope_df["label"], scope_df[pred_col])
            rows.append(
                {
                    "scope": scope_name,
                    "model": model_name,
                    "llm_called_rows": int(llm_called.sum()),
                    "llm_call_rate": float(llm_called.mean()),
                    "llm_valid_rows": int(llm_valid.sum()),
                    "llm_valid_rate": float(llm_valid.mean()),
                    "prediction_changed_vs_rag_vote_rate": float(
                        (scope_df["pred_rag_llm"] != scope_df["pred_rag_vote"]).mean()
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def build_route_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how many rows fall into each production route bucket."""
    required = {"recommended_route", "pred_linear_model_plus_rag_conf"}
    if not required.issubset(test_df.columns):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for route_name, scope_df in test_df.groupby("recommended_route", dropna=False):
        route_label = str(route_name) if pd.notna(route_name) else "missing"
        rows.append(
            {
                "recommended_route": route_label,
                "rows": int(len(scope_df)),
                "row_rate": float(len(scope_df) / len(test_df)) if len(test_df) else 0.0,
                "mean_hybrid_confidence": float(scope_df["pred_linear_model_plus_rag_conf"].mean()),
                "accuracy_if_auto_used": float(
                    accuracy_score(scope_df["label"], scope_df["pred_linear_model_plus_rag"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["recommended_route"]).reset_index(drop=True)


def _empty_rag_result(fallback_label: str) -> Dict[str, Any]:
    return {
        "label": fallback_label,
        "neighbors": [],
        "best_score": 0.0,
        "second_score": 0.0,
        "margin": 0.0,
        "label_scores": {"Primary": 0.0, "Reuse": 0.0},
    }


class RagExampleRetriever:
    """Retrieve nearest labeled evidence snippets from the training fold only."""

    def __init__(self, df: pd.DataFrame, max_features: int = 20000):
        work = df.copy().reset_index(drop=True)
        work["retrieval_text"] = work.apply(
            lambda row: build_provenance_retrieval_text(
                title=str(row.get("title", "")),
                evidence_text=str(row.get("evidence_text", "")),
            ),
            axis=1,
        )
        work["retrieval_profile"] = work.get("evidence_text", "").fillna("").astype(str).apply(provenance_profile)
        work = work[work["retrieval_text"].str.len() > 0].reset_index(drop=True)

        self.df = work
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        if work.empty:
            return

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            max_features=max_features,
        )
        self.matrix = self.vectorizer.fit_transform(work["retrieval_text"])

    def retrieve(
        self,
        title: str,
        evidence_text: str,
        top_k: int,
        min_similarity: float,
        candidate_pool: int,
        per_label_cap: int,
        exclude_paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.df.empty or self.vectorizer is None or self.matrix is None:
            return []

        query_profile = provenance_profile(evidence_text)
        query_text = build_provenance_retrieval_text(title=title, evidence_text=evidence_text)
        if not query_text:
            return []

        qv = self.vectorizer.transform([query_text])
        sims = cosine_similarity(qv, self.matrix).ravel()
        idxs = np.argsort(-sims)[: max(top_k * 2, candidate_pool * 2, 20)]

        rows: List[Dict[str, Any]] = []
        per_label_count: Dict[str, int] = {"Primary": 0, "Reuse": 0}
        for i in idxs:
            row = self.df.iloc[int(i)]
            if exclude_paper_id is not None and str(row.get("paper_id", "")) == str(exclude_paper_id):
                continue
            base_score = float(sims[int(i)])
            if base_score < min_similarity:
                continue
            candidate_profile = row["retrieval_profile"]
            hybrid_score = base_score + provenance_alignment_score(query_profile, candidate_profile)
            if hybrid_score < min_similarity:
                continue
            label = str(row["label"])
            if per_label_count.get(label, 0) >= per_label_cap:
                continue
            rows.append(
                {
                    "paper_id": str(row.get("paper_id", "")),
                    "label": label,
                    "score": hybrid_score,
                    "base_score": base_score,
                    "title": str(row.get("title", "")),
                    "snippet": str(row.get("evidence_text", ""))[:280],
                    "structured_summary": str(row.get("structured_evidence_summary", ""))[:220],
                }
            )
            per_label_count[label] = per_label_count.get(label, 0) + 1
            if len(rows) >= candidate_pool:
                break
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        return rows[:top_k]


def rag_vote_predict_with_details(
    title: str,
    evidence_text: str,
    retriever: RagExampleRetriever,
    top_k: int,
    candidate_pool: int,
    per_label_cap: int,
    min_similarity: float,
    vote_margin: float,
    fallback_label: str = "Unclear",
    exclude_paper_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return retrieval neighbors, similarity totals, and final voted label."""
    if not re.sub(r"\s+", " ", evidence_text or "").strip():
        return _empty_rag_result(fallback_label)

    neighbors = retriever.retrieve(
        title=title,
        evidence_text=evidence_text,
        top_k=top_k,
        min_similarity=min_similarity,
        candidate_pool=candidate_pool,
        per_label_cap=per_label_cap,
        exclude_paper_id=exclude_paper_id,
    )
    if not neighbors:
        return _empty_rag_result(fallback_label)

    label_scores: Dict[str, float] = {"Primary": 0.0, "Reuse": 0.0}
    for item in neighbors:
        label_scores[item["label"]] = label_scores.get(item["label"], 0.0) + float(item["score"])

    ranked = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = float(best_score - second_score)

    if best_score <= 0.0 or margin < vote_margin:
        return {
            "label": fallback_label,
            "neighbors": neighbors,
            "best_score": float(best_score),
            "second_score": float(second_score),
            "margin": margin,
            "label_scores": label_scores,
        }
    return {
        "label": best_label,
        "neighbors": neighbors,
        "best_score": float(best_score),
        "second_score": float(second_score),
        "margin": margin,
        "label_scores": label_scores,
    }


def rag_vote_predict(
    title: str,
    evidence_text: str,
    retriever: RagExampleRetriever,
    top_k: int,
    candidate_pool: int,
    per_label_cap: int,
    min_similarity: float,
    vote_margin: float,
    fallback_label: str = "Unclear",
    exclude_paper_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Predict label by summing similarity scores of retrieved neighbors.

    Why weighted voting instead of plain majority vote:
    - closer neighbors should matter more than weakly related ones
    - a small labeled set benefits from using similarity as soft evidence
    - the returned neighbor list is saved for manual error analysis
    """
    result = rag_vote_predict_with_details(
        title=title,
        evidence_text=evidence_text,
        retriever=retriever,
        top_k=top_k,
        candidate_pool=candidate_pool,
        per_label_cap=per_label_cap,
        min_similarity=min_similarity,
        vote_margin=vote_margin,
        fallback_label=fallback_label,
        exclude_paper_id=exclude_paper_id,
    )
    return str(result["label"]), list(result["neighbors"])


def _extract_accessions_for_prompt(text: str, cap: int = 20) -> str:
    acc = sorted(set(m.group(0).upper() for m in ACC_ANY.finditer(text or "")))
    if not acc:
        return "None"
    return ", ".join(acc[:cap])


def _mined_phrase_hints(mined: Dict[str, List[Dict[str, float]]], per_class: int = 8) -> Dict[str, List[str]]:
    hints: Dict[str, List[str]] = {"Primary": [], "Reuse": []}
    for cls in ["Primary", "Reuse"]:
        phrases = []
        for item in mined.get(cls, []):
            phrase = str(item.get("phrase", "")).strip()
            if phrase:
                phrases.append(phrase)
        hints[cls] = phrases[:per_class]
    return hints


def format_rag_neighbors_for_context(rag_neighbors: List[Dict[str, Any]]) -> str:
    """Render retrieved neighbors as a compact standard-RAG context block."""
    if rag_neighbors:
        return "\n".join(
            f"- label={row['label']} sim={row['score']:.3f} title={row['title'][:120]} "
            f"roles={row.get('structured_summary', '')[:140]} snippet={row.get('snippet', '')[:160]}"
            for row in rag_neighbors
        )
    return "(none)"


def build_standard_rag_context(
    accessions: str,
    structured_evidence: str,
    evidence: str,
    rag_neighbors: List[Dict[str, Any]],
) -> str:
    """Textbook RAG context: target evidence plus retrieved labeled support."""
    neighbors_text = format_rag_neighbors_for_context(rag_neighbors)
    return (
        f"Task: classify the paper as Primary or Reuse.\n"
        f"Target accessions:\n{accessions}\n\n"
        f"Structured evidence:\n{structured_evidence or '(none)'}\n\n"
        f"Target evidence:\n{evidence}\n\n"
        f"Retrieved labeled neighbors:\n{neighbors_text}"
    )


def build_rag_llm_prompt(
    accessions: str,
    evidence: str,
    structured_evidence: str,
    rag_neighbors: List[Dict[str, Any]],
    phrase_hints: Dict[str, List[str]],
) -> str:
    neighbors_text = format_rag_neighbors_for_context(rag_neighbors)

    primary_hints = ", ".join(phrase_hints.get("Primary", [])) or "(none)"
    reuse_hints = ", ".join(phrase_hints.get("Reuse", [])) or "(none)"

    return f"""You are an expert reviewer for data provenance classification in biomedical papers.

Task:
Classify this paper as:
- Primary: authors generated the dataset used in the main analysis (even if they later deposited it).
- Reuse: authors mainly reused public/external datasets.
Also identify the single most important sentence for the final label.

Decision rules:
1. Give highest weight to Methods/Data-availability/accession-adjacent evidence.
2. "we generated/sequenced/collected/recruited/library preparation" suggests Primary.
3. "downloaded/publicly available/reanalyzed GSE/obtained from GEO" means Reuse unless there is explicit evidence that the authors generated the main dataset themselves.
4. Downstream analysis, reanalysis, integration, normalization, modeling, or new statistics on public data does NOT make the dataset Primary.
5. If both Primary-like and Reuse-like evidence appear, choose Primary only when the paper itself generated a core dataset used in the main analysis.
6. Do not decide only from title style when direct evidence exists.
7. Return the strongest single sentence that best supports the final label.

Retrieved labeled neighbors (guidance only):
{neighbors_text}

Mined phrase hints:
- Primary-like: {primary_hints}
- Reuse-like: {reuse_hints}

Target accessions:
{accessions}

Structured evidence:
{structured_evidence or "(none)"}

Target evidence:
{evidence}

Keep rationale, primary_evidence, and reuse_evidence each under 20 words.
Output minified JSON only with keys exactly: label, confidence, rationale, main_decision_sentence, primary_evidence, reuse_evidence.
{{"label":"Primary|Reuse","confidence":0.0-1.0,"rationale":"short evidence-based reason","main_decision_sentence":"best single sentence","primary_evidence":"best direct primary clue or none","reuse_evidence":"best direct reuse clue or none"}}"""


def build_rag_llm_sentence_judge_prompt(
    accessions: str,
    evidence: str,
    structured_evidence: str,
    rag_neighbors: List[Dict[str, Any]],
    phrase_hints: Dict[str, List[str]],
) -> str:
    neighbors_text = format_rag_neighbors_for_context(rag_neighbors)
    primary_hints = ", ".join(phrase_hints.get("Primary", [])) or "(none)"
    reuse_hints = ", ".join(phrase_hints.get("Reuse", [])) or "(none)"

    return f"""You are reviewing biomedical paper evidence for data provenance.

Task:
1. Identify the single most important sentence for deciding Primary vs Reuse.
2. Assign that sentence a role:
   - Primary_generation
   - Primary_deposition
   - Reuse_download
   - Reuse_reanalysis
   - Mixed_conflict
   - Other_provenance
3. Predict the final label: Primary or Reuse.

Priority rules:
1. Use direct evidence near GEO/GSE/accession mentions first.
2. "downloaded from GEO", "publicly available dataset", "reanalyzed GSE", "obtained from GEO" means Reuse unless there is explicit own-data-generation evidence.
3. "we generated", "we sequenced", "we collected", "we recruited" suggest Primary only when they refer to generating the main dataset.
4. Downstream analysis, integration, normalization, modeling, or new statistics on public data does NOT make the dataset Primary.
5. "deposited/submitted under GSE..." suggests Primary only if the paper indicates the authors generated the core dataset.
6. Prefer the sentence that best describes the main dataset provenance, not a generic analysis sentence.
7. If the paper both reuses public data and also generates its own core dataset, choose Primary.

Retrieved labeled neighbors (guidance only):
{neighbors_text}

Mined phrase hints:
- Primary-like: {primary_hints}
- Reuse-like: {reuse_hints}

Target accessions:
{accessions}

Structured evidence:
{structured_evidence or "(none)"}

Target evidence:
{evidence}

Keep rationale, primary_evidence, and reuse_evidence each under 20 words.
Output minified JSON only with keys exactly:
{{"label":"Primary|Reuse","confidence":0.0-1.0,"rationale":"short evidence-based reason","main_decision_sentence":"best single sentence","main_decision_role":"Primary_generation|Primary_deposition|Reuse_download|Reuse_reanalysis|Mixed_conflict|Other_provenance","primary_evidence":"best direct primary clue or none","reuse_evidence":"best direct reuse clue or none"}}"""


def build_rag_llm_verifier_prompt(
    accessions: str,
    evidence: str,
    structured_evidence: str,
    rag_neighbors: List[Dict[str, Any]],
    phrase_hints: Dict[str, List[str]],
    base_label: str,
    rag_margin: float,
    linear_label: str,
    static_rule_label: str,
) -> str:
    """Ask the LLM to review and optionally override the retrieval vote.

    The key instruction is conservative: keep the base RAG label unless direct
    evidence from the target paper clearly contradicts it.
    """
    neighbors_text = format_rag_neighbors_for_context(rag_neighbors)

    primary_hints = ", ".join(phrase_hints.get("Primary", [])) or "(none)"
    reuse_hints = ", ".join(phrase_hints.get("Reuse", [])) or "(none)"

    return f"""You are reviewing a biomedical data-provenance classification.

Your job is to REVIEW a baseline prediction, not to guess from scratch.

Baseline signals:
- RAG vote label: {base_label}
- RAG vote margin: {rag_margin:.4f}
- Linear model label: {linear_label}
- Static rule label: {static_rule_label}

High-priority rules:
1. Default to KEEP the RAG vote unless direct evidence from the target paper clearly contradicts it.
2. "downloaded from GEO", "publicly available dataset", "reanalyzed GSE", "obtained from GEO/TCGA", or integrating multiple public datasets usually means Reuse, even if the authors later reprocessed, normalized, reannotated, or ran new statistics.
3. "we generated/sequenced/collected/recruited/library preparation" usually means Primary.
4. "data deposited/submitted under GSE..." means Primary only when the paper also indicates that the authors generated the core dataset.
5. Methods and data-availability text are stronger than general analysis wording.
6. Do not override to Primary just because the authors performed new downstream analysis on public data.
7. If both reuse and own-generation evidence appear, choose Primary when the authors generated any core dataset used in the main analysis.

Retrieved labeled neighbors:
{neighbors_text}

Mined phrase hints:
- Primary-like: {primary_hints}
- Reuse-like: {reuse_hints}

Target accessions:
{accessions}

Structured evidence:
{structured_evidence or "(none)"}

Target evidence:
{evidence}

Keep rationale, primary_evidence, and reuse_evidence each under 20 words.
Return minified JSON only with keys exactly:
{{"decision":"keep|override","label":"Primary|Reuse","confidence":0.0-1.0,"rationale":"short evidence-based reason","primary_evidence":"best direct primary clue or none","reuse_evidence":"best direct reuse clue or none"}}"""


def _infer_label_from_text(raw: str) -> Optional[str]:
    text = str(raw or "")
    labels = set(m.group(0).title() for m in re.finditer(r"(?i)\b(primary|reuse|unclear)\b", text))
    if len(labels) == 1:
        return next(iter(labels))
    return None


def call_ollama_for_eval(prompt: str, cfg: ModelingConfig) -> Dict[str, Any]:
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": cfg.llm_temperature,
            "num_predict": cfg.llm_num_predict,
        },
    }
    response = requests.post(
        cfg.ollama_url,
        json=payload,
        timeout=(cfg.llm_timeout_connect, cfg.llm_timeout_read),
    )
    response.raise_for_status()
    raw = response.json().get("response", "").strip()

    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:
                obj = None
        else:
            obj = None
    if obj is None:
        fallback_label = _infer_label_from_text(raw) or "Unclear"
        return {"label": fallback_label, "confidence": 0.51, "rationale": "No strict JSON found", "raw": raw}

    label = str(obj.get("label", "Unclear")).title()
    if label not in {"Primary", "Reuse", "Unclear"}:
        label = "Unclear"

    try:
        conf = float(obj.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    decision = str(obj.get("decision", "")).strip().lower()
    if decision not in {"keep", "override"}:
        decision = ""

    return {
        "label": label,
        "confidence": conf,
        "decision": decision,
        "rationale": str(obj.get("rationale", "")).strip(),
        "main_decision_sentence": str(obj.get("main_decision_sentence", "")).strip(),
        "main_decision_role": str(obj.get("main_decision_role", "")).strip(),
        "primary_evidence": str(obj.get("primary_evidence", "")).strip(),
        "reuse_evidence": str(obj.get("reuse_evidence", "")).strip(),
        "raw": raw,
    }


def load_article_text_from_jsonl(jsonl_path: Path) -> pd.DataFrame:
    """Load article text from JSONL and return unique paper_id -> article_text mapping."""
    rows: List[Dict[str, str]] = []
    for rec in iter_jsonl(jsonl_path):
        paper_id = rec.get("paper_id") or rec.get("pmcid") or rec.get("doi") or rec.get("id")
        if not paper_id:
            continue
        rows.append({"paper_id": str(paper_id), "article_text": extract_text_fields(rec)})

    if not rows:
        return pd.DataFrame(columns=["paper_id", "article_text"])

    text_df = pd.DataFrame(rows).drop_duplicates(subset=["paper_id"], keep="first")
    return text_df


def should_call_llm(
    llm_eval_mode: str,
    rag_label: str,
    rag_margin: float,
    linear_label: str,
    static_rule_label: str,
    recommended_route: str,
    recommended_route_reason: str,
    cfg: ModelingConfig,
) -> Tuple[bool, str]:
    """Decide whether the LLM should review this row."""
    if llm_eval_mode == "all":
        return True, "all_rows"
    if llm_eval_mode == "production_route":
        if recommended_route == "llm_review":
            return True, f"production_route|{recommended_route_reason or 'llm_review'}"
        return False, f"production_route|{recommended_route or 'skip'}"
    if llm_eval_mode == "unclear_only":
        if rag_label == "Unclear":
            return True, "rag_vote_unclear"
        return False, "skip_not_unclear"
    if llm_eval_mode == "hard_cases":
        reasons: List[str] = []
        if static_rule_label == "Unclear":
            reasons.append("static_rules_unclear")
        if static_rule_label in {"Primary", "Reuse"} and static_rule_label != rag_label:
            reasons.append("static_rules_disagree")
        if rag_label != linear_label:
            reasons.append("rag_linear_disagree")
        if rag_margin < cfg.llm_hard_margin:
            reasons.append("low_rag_margin")
        if reasons:
            return True, "|".join(reasons)
        return False, "skip_not_hard_case"
    return False, "llm_off"


def build_or_load_split(
    df: pd.DataFrame,
    cfg: ModelingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Return train/test DataFrames and the split manifest used.

    If `split_path` is provided, the split is loaded by paper_id. Otherwise a
    stratified split is created and can optionally be saved for reuse.
    """
    if "paper_id" not in df.columns:
        raise ValueError("Dataset must contain paper_id to save or load a fixed split.")

    work = df.copy()
    work["paper_id"] = work["paper_id"].astype(str)
    save_path = cfg.save_split_path or (cfg.out_dir / "heldout_split.json")

    if cfg.split_path is not None:
        manifest = json.loads(cfg.split_path.read_text(encoding="utf-8"))
        train_ids = [str(x) for x in manifest.get("train_paper_ids", [])]
        test_ids = [str(x) for x in manifest.get("test_paper_ids", [])]
        train_df = work[work["paper_id"].isin(train_ids)].copy()
        test_df = work[work["paper_id"].isin(test_ids)].copy()
        if len(train_df) != len(train_ids) or len(test_df) != len(test_ids):
            raise ValueError("Loaded split_path does not match the current dataset by paper_id.")
        return train_df, test_df, manifest

    train_df, test_df = train_test_split(
        work,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=work["label"],
    )
    manifest = {
        "dataset_rows": int(len(work)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_size": float(cfg.test_size),
        "random_state": int(cfg.random_state),
        "extraction_mode": cfg.extraction_mode,
        "train_paper_ids": train_df["paper_id"].astype(str).tolist(),
        "test_paper_ids": test_df["paper_id"].astype(str).tolist(),
        "label_counts_train": train_df["label"].value_counts().to_dict(),
        "label_counts_test": test_df["label"].value_counts().to_dict(),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_df, test_df, manifest


def build_labeled_dataset(cfg: ModelingConfig) -> pd.DataFrame:
    """Build training table with labels + evidence text.

    If JSONL is provided: merge CSV labels with JSONL articles on paper_id.
    Else: fall back to text columns inside CSV itself.
    """
    labels_df = read_csv_flex(cfg.labeled_csv_path)

    if "paper_id" not in labels_df.columns:
        raise ValueError("Labeled CSV must contain 'paper_id' for safe matching.")

    label_col = "ground_truth" if "ground_truth" in labels_df.columns else "human_label"
    if label_col not in labels_df.columns:
        raise ValueError("CSV must contain 'ground_truth' or 'human_label'.")

    labels_df = labels_df.copy()
    labels_df["paper_id"] = labels_df["paper_id"].astype(str)
    labels_df["label"] = labels_df[label_col].apply(normalize_label)
    labels_df = labels_df[labels_df["label"].isin(["Primary", "Reuse"])].copy()

    if cfg.jsonl_path is not None:
        text_df = load_article_text_from_jsonl(cfg.jsonl_path)
        text_df["paper_id"] = text_df["paper_id"].astype(str)
        merged = labels_df.merge(text_df, on="paper_id", how="inner")

        dropped = len(labels_df) - len(merged)
        if dropped > 0:
            print(f"[info] Dropped {dropped} labeled rows not found in JSONL by paper_id.")

        merged["evidence_text"] = merged["article_text"].apply(
            lambda text: build_evidence_text(
                text,
                extraction_mode=cfg.extraction_mode,
                max_chars=cfg.max_evidence_chars,
                win_before=cfg.win_before,
                win_after=cfg.win_after,
            )
        )
        return enrich_with_structured_evidence(merged)

    combined = (
        labels_df.get("title", "").fillna("").astype(str)
        + " "
        + labels_df.get("abstract", "").fillna("").astype(str)
        + " "
        + labels_df.get("full_text", "").fillna("").astype(str)
    )
    labels_df["evidence_text"] = combined.apply(
        lambda text: build_evidence_text(
            text,
            extraction_mode=cfg.extraction_mode,
            max_chars=cfg.max_evidence_chars,
            win_before=cfg.win_before,
            win_after=cfg.win_after,
        )
    )
    return enrich_with_structured_evidence(labels_df)


def evaluate_methods_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: ModelingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[Dict[str, float]]], str]:
    """Evaluate all benchmark methods on one train/test split."""
    test_df = test_df.copy()
    test_df["pred_static_rules"] = test_df["evidence_text"].apply(classic_heuristic)

    mined = mine_top_phrases(
        texts=train_df["evidence_text"],
        labels=train_df["label"],
        top_k=cfg.top_k_phrases,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
        max_features=cfg.max_features,
    )
    template_rules = build_template_rules(mined)
    test_df["pred_mined_templates"] = test_df["evidence_text"].apply(lambda x: template_rule_predict(x, template_rules))

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, max_features=20000)
    X_train = vec.fit_transform(train_df["evidence_text"].fillna(""))
    X_test = vec.transform(test_df["evidence_text"].fillna(""))

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_state)
    clf.fit(X_train, train_df["label"])
    test_df["pred_linear_model"] = clf.predict(X_test)
    linear_probs = _probability_frame(clf, X_test).reset_index(drop=True)
    test_df["pred_linear_model_primary_prob"] = linear_probs["prob_Primary"].to_numpy()
    test_df["pred_linear_model_reuse_prob"] = linear_probs["prob_Reuse"].to_numpy()
    test_df["pred_linear_model_conf"] = np.maximum(
        test_df["pred_linear_model_primary_prob"],
        test_df["pred_linear_model_reuse_prob"],
    )

    retriever = RagExampleRetriever(train_df, max_features=cfg.max_features)
    train_rag_outputs = train_df.apply(
        lambda row: rag_vote_predict_with_details(
            title=str(row.get("title", "")),
            evidence_text=str(row.get("evidence_text", "")),
            retriever=retriever,
            top_k=cfg.rag_top_k,
            candidate_pool=cfg.rag_candidate_pool,
            per_label_cap=cfg.rag_per_label_cap,
            min_similarity=cfg.rag_min_similarity,
            vote_margin=cfg.rag_vote_margin,
            exclude_paper_id=str(row.get("paper_id", "")),
        ),
        axis=1,
    )
    train_profiles = train_df["evidence_text"].fillna("").astype(str).apply(provenance_profile)
    train_rag_dense = np.asarray(
        [rag_feature_row(result, profile) for result, profile in zip(train_rag_outputs, train_profiles)],
        dtype=float,
    )
    rag_outputs = test_df.apply(
        lambda row: rag_vote_predict_with_details(
            title=str(row.get("title", "")),
            evidence_text=str(row.get("evidence_text", "")),
            retriever=retriever,
            top_k=cfg.rag_top_k,
            candidate_pool=cfg.rag_candidate_pool,
            per_label_cap=cfg.rag_per_label_cap,
            min_similarity=cfg.rag_min_similarity,
            vote_margin=cfg.rag_vote_margin,
        ),
        axis=1,
    )
    test_profiles = test_df["evidence_text"].fillna("").astype(str).apply(provenance_profile)
    test_rag_dense = np.asarray(
        [rag_feature_row(result, profile) for result, profile in zip(rag_outputs, test_profiles)],
        dtype=float,
    )
    X_train_hybrid = hstack([X_train, csr_matrix(train_rag_dense)], format="csr")
    X_test_hybrid = hstack([X_test, csr_matrix(test_rag_dense)], format="csr")
    clf_hybrid = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_state)
    clf_hybrid.fit(X_train_hybrid, train_df["label"])
    test_df["pred_linear_model_plus_rag"] = clf_hybrid.predict(X_test_hybrid)
    hybrid_probs = _probability_frame(clf_hybrid, X_test_hybrid).reset_index(drop=True)
    test_df["pred_linear_model_plus_rag_primary_prob"] = hybrid_probs["prob_Primary"].to_numpy()
    test_df["pred_linear_model_plus_rag_reuse_prob"] = hybrid_probs["prob_Reuse"].to_numpy()
    test_df["pred_linear_model_plus_rag_conf"] = np.maximum(
        test_df["pred_linear_model_plus_rag_primary_prob"],
        test_df["pred_linear_model_plus_rag_reuse_prob"],
    )

    rag_labels = [str(item["label"]) for item in rag_outputs]
    rag_neighbors_per_row = [list(item["neighbors"]) for item in rag_outputs]
    rag_margins = [float(item["margin"]) for item in rag_outputs]
    rag_primary_scores = [float(item["label_scores"].get("Primary", 0.0)) for item in rag_outputs]
    rag_reuse_scores = [float(item["label_scores"].get("Reuse", 0.0)) for item in rag_outputs]
    test_df["pred_rag_vote"] = rag_labels
    test_df["pred_rag_vote_margin"] = rag_margins
    test_df["pred_rag_vote_primary_score"] = rag_primary_scores
    test_df["pred_rag_vote_reuse_score"] = rag_reuse_scores
    test_df["pred_rag_neighbor_count"] = [len(neighbors) for neighbors in rag_neighbors_per_row]
    test_df["rag_neighbors"] = [
        json.dumps(
            [
                {
                    "paper_id": n["paper_id"],
                    "label": n["label"],
                    "score": round(float(n["score"]), 4),
                    "base_score": round(float(n.get("base_score", 0.0)), 4),
                    "title": n["title"],
                    "structured_summary": n.get("structured_summary", ""),
                    "snippet": n.get("snippet", ""),
                }
                for n in neighbors
            ],
            ensure_ascii=False,
        )
        for neighbors in rag_neighbors_per_row
    ]
    test_df["standard_rag_context"] = [
        build_standard_rag_context(
            accessions=_extract_accessions_for_prompt(str(row.get("evidence_text", ""))),
            structured_evidence=str(row.get("structured_evidence_summary", ""))[:1600],
            evidence=str(row.get("evidence_text", ""))[:1800],
            rag_neighbors=rag_neighbors_per_row[row_idx][: max(1, cfg.llm_rag_top_k)],
        )
        for row_idx, (_, row) in enumerate(test_df.iterrows())
    ]
    test_df["pred_rules_plus_rag"] = np.where(
        test_df["pred_static_rules"] != "Unclear",
        test_df["pred_static_rules"],
        test_df["pred_rag_vote"],
    )
    route_outputs = [
        recommend_review_route(
            hybrid_label=str(row["pred_linear_model_plus_rag"]),
            hybrid_conf=float(row["pred_linear_model_plus_rag_conf"]),
            rag_label=str(row["pred_rag_vote"]),
            rag_margin=float(row["pred_rag_vote_margin"]),
            static_rule_label=str(row["pred_static_rules"]),
            profile=profile,
            cfg=cfg,
        )
        for (_, row), profile in zip(test_df.iterrows(), test_profiles)
    ]
    test_df["recommended_route"] = [route for route, _ in route_outputs]
    test_df["recommended_route_reason"] = [reason for _, reason in route_outputs]

    if cfg.llm_eval_mode != "off":
        phrase_hints = _mined_phrase_hints(mined, per_class=8)
        call_budget = cfg.llm_eval_max_rows if cfg.llm_eval_max_rows > 0 else len(test_df)
        call_count = 0
        llm_labels: List[str] = []
        llm_conf: List[float] = []
        llm_reason: List[str] = []
        llm_called: List[int] = []
        llm_valid: List[int] = []
        llm_decision: List[str] = []
        llm_gate_reason: List[str] = []
        llm_model_label: List[str] = []

        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            fallback_label = str(row["pred_rag_vote"])
            should_call, gate_reason = should_call_llm(
                llm_eval_mode=cfg.llm_eval_mode,
                rag_label=fallback_label,
                rag_margin=float(row["pred_rag_vote_margin"]),
                linear_label=str(row["pred_linear_model_plus_rag"]),
                static_rule_label=str(row["pred_static_rules"]),
                recommended_route=str(row.get("recommended_route", "")),
                recommended_route_reason=str(row.get("recommended_route_reason", "")),
                cfg=cfg,
            )
            if should_call and call_count < call_budget:
                if cfg.llm_strategy == "verify_override":
                    prompt = build_rag_llm_verifier_prompt(
                        accessions=_extract_accessions_for_prompt(str(row.get("evidence_text", ""))),
                        evidence=str(row.get("evidence_text", ""))[:1800],
                        structured_evidence=str(row.get("structured_evidence_summary", ""))[:1600],
                        rag_neighbors=rag_neighbors_per_row[row_idx][: max(1, cfg.llm_rag_top_k)],
                        phrase_hints=phrase_hints,
                        base_label=fallback_label,
                        rag_margin=float(row["pred_rag_vote_margin"]),
                        linear_label=str(row["pred_linear_model_plus_rag"]),
                        static_rule_label=str(row["pred_static_rules"]),
                    )
                else:
                    prompt = build_rag_llm_prompt(
                        accessions=_extract_accessions_for_prompt(str(row.get("evidence_text", ""))),
                        evidence=str(row.get("evidence_text", ""))[:1800],
                        structured_evidence=str(row.get("structured_evidence_summary", ""))[:1600],
                        rag_neighbors=rag_neighbors_per_row[row_idx][: max(1, cfg.llm_rag_top_k)],
                        phrase_hints=phrase_hints,
                    )
                try:
                    out = call_ollama_for_eval(prompt, cfg)
                    if str(out.get("rationale", "")).strip() == "No strict JSON found":
                        llm_labels.append(fallback_label)
                        llm_conf.append(0.5)
                        llm_reason.append("No strict JSON found; fallback to rag-vote")
                        llm_valid.append(0)
                        llm_decision.append("fallback_invalid_json")
                        llm_model_label.append("")
                    else:
                        parsed_label = str(out["label"])
                        parsed_conf = float(out["confidence"])
                        if cfg.llm_strategy == "verify_override":
                            decision = str(out.get("decision", "")).strip().lower()
                            if decision == "override":
                                final_label = parsed_label
                                final_decision = "override"
                            elif parsed_label != fallback_label and parsed_conf >= 0.6:
                                final_label = parsed_label
                                final_decision = "override_inferred_from_label"
                            else:
                                final_label = fallback_label
                                final_decision = "keep"
                        else:
                            final_label = parsed_label
                            final_decision = "classify"
                        llm_labels.append(final_label)
                        llm_conf.append(parsed_conf)
                        llm_reason.append(str(out["rationale"]))
                        llm_valid.append(1)
                        llm_decision.append(final_decision)
                        llm_model_label.append(parsed_label)
                except Exception as exc:
                    llm_labels.append(fallback_label)
                    llm_conf.append(0.5)
                    llm_reason.append(f"LLM error; fallback to rag-vote: {exc}")
                    llm_valid.append(0)
                    llm_decision.append("fallback_error")
                    llm_model_label.append("")
                llm_called.append(1)
                llm_gate_reason.append(gate_reason)
                call_count += 1
            else:
                llm_labels.append(fallback_label)
                llm_conf.append(0.5)
                llm_reason.append("rag-vote-only")
                llm_called.append(0)
                llm_valid.append(0)
                llm_decision.append("skip")
                llm_gate_reason.append(gate_reason)
                llm_model_label.append("")

        test_df["pred_rag_llm"] = llm_labels
        test_df["pred_rag_llm_conf"] = llm_conf
        test_df["pred_rag_llm_reason"] = llm_reason
        test_df["pred_rag_llm_called"] = llm_called
        test_df["pred_rag_llm_valid"] = llm_valid
        test_df["pred_rag_llm_decision"] = llm_decision
        test_df["pred_rag_llm_gate_reason"] = llm_gate_reason
        test_df["pred_rag_llm_model_label"] = llm_model_label

    rows = []
    for name, col in [
        ("static_rules", "pred_static_rules"),
        ("mined_template_rules", "pred_mined_templates"),
        ("linear_model", "pred_linear_model"),
        ("linear_model_plus_rag", "pred_linear_model_plus_rag"),
        ("rag_vote", "pred_rag_vote"),
        ("rules_plus_rag", "pred_rules_plus_rag"),
    ]:
        m = evaluate_predictions(test_df["label"], test_df[col])
        m["model"] = name
        rows.append(m)
    if cfg.llm_eval_mode != "off":
        m = evaluate_predictions(test_df["label"], test_df["pred_rag_llm"])
        m["model"] = "rag_llm"
        rows.append(m)

    metrics_df = pd.DataFrame(rows).set_index("model")
    report = classification_report(test_df["label"], test_df["pred_linear_model"], digits=4)
    return metrics_df, test_df, mined, report


def cross_validate_extraction_modes(
    cfg: ModelingConfig,
    extraction_modes: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare extraction modes with stratified cross-validation.

    This gives a more stable view than one lucky/unlucky train/test split,
    which matters a lot when the labeled set is still small.
    """
    modes = extraction_modes or ["accession_windows", "section_aware"]
    modes = list(dict.fromkeys(modes))
    fold_rows: List[Dict[str, Any]] = []

    for mode in modes:
        mode_cfg = ModelingConfig(**{**cfg.__dict__, "extraction_mode": mode})
        if not mode_cfg.llm_eval_in_cv:
            mode_cfg.llm_eval_mode = "off"
        df = build_labeled_dataset(mode_cfg)
        splitter = StratifiedKFold(n_splits=mode_cfg.cv_folds, shuffle=True, random_state=mode_cfg.random_state)

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(df, df["label"]), start=1):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            metrics_df, _, _, _ = evaluate_methods_on_split(train_df, test_df, mode_cfg)

            for model_name, row in metrics_df.reset_index().iterrows():
                fold_rows.append(
                    {
                        "extraction_mode": mode,
                        "fold": fold_id,
                        "model": row["model"],
                        "accuracy_all": float(row["accuracy_all"]),
                        "accuracy_binary_on_covered": float(row["accuracy_binary_on_covered"]),
                        "coverage_binary_pred": float(row["coverage_binary_pred"]),
                        "unclear_rate": float(row["unclear_rate"]),
                        "n": int(row["n"]),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby(["extraction_mode", "model"], as_index=False)
        .agg(
            accuracy_all_mean=("accuracy_all", "mean"),
            accuracy_all_std=("accuracy_all", "std"),
            accuracy_binary_mean=("accuracy_binary_on_covered", "mean"),
            coverage_mean=("coverage_binary_pred", "mean"),
            unclear_rate_mean=("unclear_rate", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(["accuracy_all_mean", "accuracy_binary_mean"], ascending=False)
        .reset_index(drop=True)
    )
    return summary_df, fold_df


def model_catalog() -> pd.DataFrame:
    """Describe every model row written to model_comparison.csv.

    This is documentation as data: the CSV makes it clear which methods use
    rules, supervised training labels, retrieval, and/or an LLM.
    """
    rows = [
        {
            "model": "static_rules",
            "uses_rules": True,
            "uses_training_labels": False,
            "uses_rag": False,
            "uses_llm": False,
            "input": "extracted evidence text",
            "description": "Hand-written provenance rules for obvious generated-data or reuse wording.",
        },
        {
            "model": "mined_template_rules",
            "uses_rules": True,
            "uses_training_labels": True,
            "uses_rag": False,
            "uses_llm": False,
            "input": "extracted evidence text",
            "description": "Regex-like templates built from class-associated phrases mined only from the training fold.",
        },
        {
            "model": "linear_model",
            "uses_rules": False,
            "uses_training_labels": True,
            "uses_rag": False,
            "uses_llm": False,
            "input": "extracted evidence text",
            "description": "TF-IDF features plus balanced logistic regression trained on the training fold.",
        },
        {
            "model": "linear_model_plus_rag",
            "uses_rules": False,
            "uses_training_labels": True,
            "uses_rag": True,
            "uses_llm": False,
            "input": "extracted evidence text plus provenance-aware RAG score features",
            "description": "Balanced logistic regression on TF-IDF evidence text augmented with provenance-aware RAG vote scores, margins, neighbor counts, and cue counts.",
        },
        {
            "model": "rag_vote",
            "uses_rules": False,
            "uses_training_labels": True,
            "uses_rag": True,
            "uses_llm": False,
            "input": "title plus extracted evidence text",
            "description": "Retrieve nearest labeled training examples and predict by similarity-weighted label vote.",
        },
        {
            "model": "rules_plus_rag",
            "uses_rules": True,
            "uses_training_labels": True,
            "uses_rag": True,
            "uses_llm": False,
            "input": "extracted evidence text, with RAG fallback",
            "description": "Use static rules when confident; otherwise fall back to rag_vote.",
        },
        {
            "model": "rag_llm",
            "uses_rules": False,
            "uses_training_labels": True,
            "uses_rag": True,
            "uses_llm": True,
            "input": "extracted evidence text plus retrieved labeled examples",
            "description": "Optional Ollama path. In `--llm-strategy classify` it is the standard textbook RAG baseline (retrieve examples then let the LLM classify). In `--llm-strategy verify_override` it is a reviewer-style variant that can keep or override the RAG vote.",
        },
    ]
    return pd.DataFrame(rows)


def evidence_sensitivity_variants() -> List[Dict[str, Any]]:
    """Evidence variants used to test whether focused evidence helps.

    Window lengths are character counts, not token counts. The default variant
    matches the main pipeline: 350 characters before a keyword/accession and
    900 characters after it, capped at 2200 total evidence characters.
    """
    return [
        {
            "evidence_variant": "title_only",
            "extraction_mode": "title_only",
            "win_before": 0,
            "win_after": 0,
            "max_evidence_chars": 0,
            "description": "No article evidence; title text only.",
        },
        {
            "evidence_variant": "accession_short",
            "extraction_mode": "accession_windows",
            "win_before": 150,
            "win_after": 350,
            "max_evidence_chars": 1000,
            "description": "Short accession/provenance keyword windows.",
        },
        {
            "evidence_variant": "accession_default",
            "extraction_mode": "accession_windows",
            "win_before": 350,
            "win_after": 900,
            "max_evidence_chars": 2200,
            "description": "Default accession/provenance keyword windows.",
        },
        {
            "evidence_variant": "accession_wide",
            "extraction_mode": "accession_windows",
            "win_before": 700,
            "win_after": 1400,
            "max_evidence_chars": 3500,
            "description": "Wider accession/provenance keyword windows.",
        },
        {
            "evidence_variant": "section_aware_default",
            "extraction_mode": "section_aware",
            "win_before": 350,
            "win_after": 900,
            "max_evidence_chars": 2200,
            "description": "Best-scoring Methods/Data Availability style spans.",
        },
        {
            "evidence_variant": "full_text_12000",
            "extraction_mode": "full_text_truncated",
            "win_before": 0,
            "win_after": 0,
            "max_evidence_chars": 12000,
            "description": "Large front-truncated article text baseline.",
        },
    ]


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a string Series even when an optional text column is missing."""
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def apply_evidence_variant(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """Return a copy of df with evidence_text rebuilt for one sensitivity variant."""
    work = df.copy()
    mode = spec["extraction_mode"]
    if mode == "title_only":
        work["evidence_text"] = _text_series(work, "title")
    elif mode == "full_text_truncated":
        source = _text_series(work, "article_text") if "article_text" in work.columns else _text_series(work, "evidence_text")
        work["evidence_text"] = source.fillna("").astype(str).str.slice(0, int(spec["max_evidence_chars"]))
    else:
        source = _text_series(work, "article_text") if "article_text" in work.columns else _text_series(work, "evidence_text")
        work["evidence_text"] = source.fillna("").astype(str).apply(
            lambda text: build_evidence_text(
                text,
                extraction_mode=mode,
                max_chars=int(spec["max_evidence_chars"]),
                win_before=int(spec["win_before"]),
                win_after=int(spec["win_after"]),
            )
        )
    work["evidence_chars"] = work["evidence_text"].fillna("").astype(str).str.len()
    return enrich_with_structured_evidence(work)


def evaluate_linear_and_rag_cv(
    df: pd.DataFrame,
    cfg: ModelingConfig,
    spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Evaluate the strongest non-LLM models for one evidence variant."""
    rows: List[Dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(df, df["label"]), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, max_features=cfg.max_features)
        X_train = vec.fit_transform(train_df["evidence_text"].fillna(""))
        X_test = vec.transform(test_df["evidence_text"].fillna(""))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_state)
        clf.fit(X_train, train_df["label"])
        linear_pred = pd.Series(clf.predict(X_test), index=test_df.index)

        retriever = RagExampleRetriever(train_df, max_features=cfg.max_features)
        train_rag_outputs = train_df.apply(
            lambda row: rag_vote_predict_with_details(
                title=str(row.get("title", "")),
                evidence_text=str(row.get("evidence_text", "")),
                retriever=retriever,
                top_k=cfg.rag_top_k,
                candidate_pool=cfg.rag_candidate_pool,
                per_label_cap=cfg.rag_per_label_cap,
                min_similarity=cfg.rag_min_similarity,
                vote_margin=cfg.rag_vote_margin,
                exclude_paper_id=str(row.get("paper_id", "")),
            ),
            axis=1,
        )
        train_profiles = train_df["evidence_text"].fillna("").astype(str).apply(provenance_profile)
        train_rag_dense = np.asarray(
            [rag_feature_row(result, profile) for result, profile in zip(train_rag_outputs, train_profiles)],
            dtype=float,
        )
        rag_outputs = test_df.apply(
            lambda row: rag_vote_predict_with_details(
                title=str(row.get("title", "")),
                evidence_text=str(row.get("evidence_text", "")),
                retriever=retriever,
                top_k=cfg.rag_top_k,
                candidate_pool=cfg.rag_candidate_pool,
                per_label_cap=cfg.rag_per_label_cap,
                min_similarity=cfg.rag_min_similarity,
                vote_margin=cfg.rag_vote_margin,
            ),
            axis=1,
        )
        rag_pred = pd.Series([str(item["label"]) for item in rag_outputs], index=test_df.index)
        test_profiles = test_df["evidence_text"].fillna("").astype(str).apply(provenance_profile)
        test_rag_dense = np.asarray(
            [rag_feature_row(result, profile) for result, profile in zip(rag_outputs, test_profiles)],
            dtype=float,
        )
        X_train_hybrid = hstack([X_train, csr_matrix(train_rag_dense)], format="csr")
        X_test_hybrid = hstack([X_test, csr_matrix(test_rag_dense)], format="csr")
        hybrid_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_state)
        hybrid_clf.fit(X_train_hybrid, train_df["label"])
        hybrid_pred = pd.Series(hybrid_clf.predict(X_test_hybrid), index=test_df.index)

        for model_name, pred in [
            ("linear_model", linear_pred),
            ("linear_model_plus_rag", hybrid_pred),
            ("rag_vote", rag_pred),
        ]:
            metrics = evaluate_predictions(test_df["label"], pd.Series(pred, index=test_df.index))
            rows.append(
                {
                    "evidence_variant": spec["evidence_variant"],
                    "description": spec["description"],
                    "extraction_mode": spec["extraction_mode"],
                    "win_before": int(spec["win_before"]),
                    "win_after": int(spec["win_after"]),
                    "max_evidence_chars": int(spec["max_evidence_chars"]),
                    "mean_evidence_chars": float(df["evidence_chars"].mean()),
                    "fold": fold_id,
                    "model": model_name,
                    **metrics,
                }
            )
    return rows


def evidence_window_sensitivity(cfg: ModelingConfig, base_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate whether focused evidence and window length matter."""
    fold_rows: List[Dict[str, Any]] = []
    for spec in evidence_sensitivity_variants():
        variant_df = apply_evidence_variant(base_df, spec)
        fold_rows.extend(evaluate_linear_and_rag_cv(variant_df, cfg, spec))

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby(
            [
                "evidence_variant",
                "description",
                "extraction_mode",
                "win_before",
                "win_after",
                "max_evidence_chars",
                "model",
            ],
            as_index=False,
        )
        .agg(
            mean_evidence_chars=("mean_evidence_chars", "mean"),
            accuracy_all_mean=("accuracy_all", "mean"),
            accuracy_all_std=("accuracy_all", "std"),
            accuracy_binary_mean=("accuracy_binary_on_covered", "mean"),
            coverage_mean=("coverage_binary_pred", "mean"),
            unclear_rate_mean=("unclear_rate", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(["accuracy_all_mean", "accuracy_binary_mean"], ascending=False)
        .reset_index(drop=True)
    )
    return summary_df, fold_df


def train_and_compare(cfg: ModelingConfig) -> None:
    """Run one deterministic benchmark split and save all evaluation artifacts.

    Reproducibility choices:
    - one explicit train/test split controlled by `random_state`
    - phrase mining, linear model training, and RAG retrieval all use only train rows
    - test outputs include retrieved neighbors so mistakes can be audited later
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_labeled_dataset(cfg)

    if df.empty:
        raise ValueError("No matched/usable rows found after label filtering and optional JSONL merge.")

    train_df, test_df, split_manifest = build_or_load_split(df, cfg)
    metrics_df, test_df, mined, rep = evaluate_methods_on_split(train_df, test_df, cfg)
    metrics_path = cfg.out_dir / "model_comparison.csv"
    metrics_df.reset_index().to_csv(metrics_path, index=False)

    catalog_path = cfg.out_dir / "model_catalog.csv"
    model_catalog().to_csv(catalog_path, index=False)

    preds_path = cfg.out_dir / "test_predictions.csv"
    cols = [
        "paper_id",
        "label",
        "title",
        "evidence_text",
        "structured_evidence_summary",
        "structured_evidence_json",
        "standard_rag_context",
        "pred_static_rules",
        "pred_mined_templates",
        "pred_linear_model",
        "pred_linear_model_conf",
        "pred_linear_model_primary_prob",
        "pred_linear_model_reuse_prob",
        "pred_linear_model_plus_rag",
        "pred_linear_model_plus_rag_conf",
        "pred_linear_model_plus_rag_primary_prob",
        "pred_linear_model_plus_rag_reuse_prob",
        "pred_rag_vote",
        "pred_rag_vote_margin",
        "pred_rag_vote_primary_score",
        "pred_rag_vote_reuse_score",
        "pred_rag_neighbor_count",
        "recommended_route",
        "recommended_route_reason",
        "pred_rules_plus_rag",
        "pred_rag_llm",
        "pred_rag_llm_conf",
        "pred_rag_llm_reason",
        "pred_rag_llm_called",
        "pred_rag_llm_valid",
        "pred_rag_llm_decision",
        "pred_rag_llm_gate_reason",
        "pred_rag_llm_model_label",
        "rag_neighbors",
    ]
    existing_cols = [c for c in cols if c in test_df.columns]
    test_df[existing_cols].to_csv(preds_path, index=False)

    standard_rag_preview_path = cfg.out_dir / "standard_rag_preview.csv"
    preview_cols = [
        "paper_id",
        "label",
        "title",
        "structured_evidence_summary",
        "standard_rag_context",
        "pred_rag_vote",
        "pred_rag_llm",
    ]
    existing_preview_cols = [c for c in preview_cols if c in test_df.columns]
    test_df[existing_preview_cols].to_csv(standard_rag_preview_path, index=False)

    mined_path = cfg.out_dir / "mined_phrases.json"
    with mined_path.open("w", encoding="utf-8") as f:
        json.dump(mined, f, ensure_ascii=False, indent=2)

    report_path = cfg.out_dir / "linear_model_report.txt"
    report_path.write_text(rep, encoding="utf-8")

    llm_summary_df = build_llm_evaluation_summary(test_df)
    llm_summary_path = cfg.out_dir / "llm_evaluation_summary.csv"
    if not llm_summary_df.empty:
        llm_summary_df.to_csv(llm_summary_path, index=False)
    route_summary_df = build_route_summary(test_df)
    route_summary_path = cfg.out_dir / "route_summary.csv"
    if not route_summary_df.empty:
        route_summary_df.to_csv(route_summary_path, index=False)

    split_manifest_path = cfg.out_dir / "heldout_split.json"
    split_manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    cv_summary_df, cv_fold_df = cross_validate_extraction_modes(cfg)
    cv_summary_path = cfg.out_dir / "cv_model_comparison.csv"
    cv_folds_path = cfg.out_dir / "cv_fold_metrics.csv"
    cv_summary_df.to_csv(cv_summary_path, index=False)
    cv_fold_df.to_csv(cv_folds_path, index=False)

    sensitivity_summary_df, sensitivity_fold_df = evidence_window_sensitivity(cfg, df)
    sensitivity_summary_path = cfg.out_dir / "evidence_window_sensitivity.csv"
    sensitivity_folds_path = cfg.out_dir / "evidence_window_sensitivity_folds.csv"
    sensitivity_summary_df.to_csv(sensitivity_summary_path, index=False)
    sensitivity_fold_df.to_csv(sensitivity_folds_path, index=False)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {catalog_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {mined_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {split_manifest_path}")
    if not llm_summary_df.empty:
        print(f"Saved: {llm_summary_path}")
    if not route_summary_df.empty:
        print(f"Saved: {route_summary_path}")
    print(f"Saved: {cv_summary_path}")
    print(f"Saved: {cv_folds_path}")
    print(f"Saved: {sensitivity_summary_path}")
    print(f"Saved: {sensitivity_folds_path}")
    print("\n=== Model comparison ===")
    print(metrics_df)
    print("\n=== Cross-validation summary ===")
    print(cv_summary_df.head(10))
    print("\n=== Evidence window sensitivity ===")
    print(sensitivity_summary_df.head(12))




def run_notebook(
    labeled_csv_path: str,
    jsonl_path: Optional[str] = None,
    out_dir: str = "outputs_evidence_modeling",
    test_size: float = 0.2,
    random_state: int = 42,
    top_k_phrases: int = 40,
    ngram_min: int = 1,
    ngram_max: int = 3,
    min_df: int = 2,
    max_features: int = 20000,
    extraction_mode: str = "accession_windows",
    win_before: int = 350,
    win_after: int = 900,
    max_evidence_chars: int = 2200,
    cv_folds: int = 5,
    rag_top_k: int = 8,
    rag_candidate_pool: int = 12,
    rag_per_label_cap: int = 4,
    rag_min_similarity: float = 0.01,
    rag_vote_margin: float = 0.0,
    llm_eval_mode: str = "off",
    llm_strategy: str = "verify_override",
    llm_hard_margin: float = 0.05,
    llm_eval_max_rows: int = 0,
    llm_eval_in_cv: bool = False,
    ollama_url: str = "http://localhost:11434/api/generate",
    ollama_model: str = "llama3",
    llm_temperature: float = 0.0,
    llm_num_predict: int = 120,
    llm_rag_top_k: int = 3,
    llm_timeout_connect: int = 10,
    llm_timeout_read: int = 120,
    route_auto_accept_threshold: float = 0.85,
    route_llm_review_threshold: float = 0.60,
    split_path: Optional[str] = None,
    save_split_path: Optional[str] = None,
) -> pd.DataFrame:
    """Notebook-friendly wrapper for systematic evidence modeling.

    This wrapper mirrors CLI behavior but accepts Python variables directly and
    returns the main comparison table as a DataFrame for immediate notebook use.

    Example:
        metrics = run_notebook(
            labeled_csv_path="manual_ground_truth_with_GSE_links_REFRESHED.csv",
            jsonl_path="pmc_gse_articles_clean.jsonl",
        )
        metrics
    """
    cfg = ModelingConfig(
        labeled_csv_path=Path(labeled_csv_path),
        jsonl_path=Path(jsonl_path) if jsonl_path else None,
        out_dir=Path(out_dir),
        test_size=test_size,
        random_state=random_state,
        top_k_phrases=top_k_phrases,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=min_df,
        max_features=max_features,
        extraction_mode=extraction_mode,
        win_before=win_before,
        win_after=win_after,
        max_evidence_chars=max_evidence_chars,
        cv_folds=cv_folds,
        rag_top_k=rag_top_k,
        rag_candidate_pool=rag_candidate_pool,
        rag_per_label_cap=rag_per_label_cap,
        rag_min_similarity=rag_min_similarity,
        rag_vote_margin=rag_vote_margin,
        llm_eval_mode=llm_eval_mode,
        llm_strategy=llm_strategy,
        llm_hard_margin=llm_hard_margin,
        llm_eval_max_rows=llm_eval_max_rows,
        llm_eval_in_cv=llm_eval_in_cv,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        llm_temperature=llm_temperature,
        llm_num_predict=llm_num_predict,
        llm_rag_top_k=llm_rag_top_k,
        llm_timeout_connect=llm_timeout_connect,
        llm_timeout_read=llm_timeout_read,
        route_auto_accept_threshold=route_auto_accept_threshold,
        route_llm_review_threshold=route_llm_review_threshold,
        split_path=Path(split_path) if split_path else None,
        save_split_path=Path(save_split_path) if save_split_path else None,
    )

    train_and_compare(cfg)
    return pd.read_csv(cfg.out_dir / "model_comparison.csv")


def parse_args() -> ModelingConfig:
    p = argparse.ArgumentParser(description="Train and compare systematic evidence models")
    p.add_argument("--labeled-csv-path", type=Path, required=True)
    p.add_argument("--jsonl-path", type=Path, default=None, help="Optional article JSONL matched with CSV by paper_id")
    p.add_argument("--out-dir", type=Path, default=Path("outputs_evidence_modeling"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--top-k-phrases", type=int, default=40)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=3)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-features", type=int, default=20000)
    p.add_argument("--extraction-mode", choices=["accession_windows", "section_aware"], default="accession_windows")
    p.add_argument("--win-before", type=int, default=350, help="Characters before each accession/provenance hit")
    p.add_argument("--win-after", type=int, default=900, help="Characters after each accession/provenance hit")
    p.add_argument("--max-evidence-chars", type=int, default=2200, help="Maximum extracted evidence characters per article")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--rag-top-k", type=int, default=8)
    p.add_argument("--rag-candidate-pool", type=int, default=12)
    p.add_argument("--rag-per-label-cap", type=int, default=4)
    p.add_argument("--rag-min-similarity", type=float, default=0.01)
    p.add_argument("--rag-vote-margin", type=float, default=0.0)
    p.add_argument("--llm-eval-mode", choices=["off", "unclear_only", "all", "hard_cases", "production_route"], default="off")
    p.add_argument("--llm-strategy", choices=["classify", "verify_override"], default="verify_override")
    p.add_argument("--llm-hard-margin", type=float, default=0.05, help="Call verifier on low-margin RAG rows in hard_cases mode")
    p.add_argument("--llm-eval-max-rows", type=int, default=0, help="0 means no cap")
    p.add_argument("--llm-eval-in-cv", action="store_true", help="also run LLM calls in cross-validation")
    p.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-num-predict", type=int, default=120)
    p.add_argument("--llm-rag-top-k", type=int, default=3)
    p.add_argument("--llm-timeout-connect", type=int, default=10)
    p.add_argument("--llm-timeout-read", type=int, default=120)
    p.add_argument("--route-auto-accept-threshold", type=float, default=0.85, help="Confidence threshold for direct auto-accept of the hybrid classifier")
    p.add_argument("--route-llm-review-threshold", type=float, default=0.60, help="Below this hybrid confidence, route to human review instead of LLM review")
    p.add_argument("--split-path", type=Path, default=None, help="Optional saved held-out split manifest to reuse")
    p.add_argument("--save-split-path", type=Path, default=None, help="Optional path to save a newly created held-out split manifest")
    args = p.parse_args()

    return ModelingConfig(
        labeled_csv_path=args.labeled_csv_path,
        jsonl_path=args.jsonl_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        top_k_phrases=args.top_k_phrases,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
        extraction_mode=args.extraction_mode,
        win_before=args.win_before,
        win_after=args.win_after,
        max_evidence_chars=args.max_evidence_chars,
        cv_folds=args.cv_folds,
        rag_top_k=args.rag_top_k,
        rag_candidate_pool=args.rag_candidate_pool,
        rag_per_label_cap=args.rag_per_label_cap,
        rag_min_similarity=args.rag_min_similarity,
        rag_vote_margin=args.rag_vote_margin,
        llm_eval_mode=args.llm_eval_mode,
        llm_strategy=args.llm_strategy,
        llm_hard_margin=args.llm_hard_margin,
        llm_eval_max_rows=args.llm_eval_max_rows,
        llm_eval_in_cv=args.llm_eval_in_cv,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        llm_temperature=args.llm_temperature,
        llm_num_predict=args.llm_num_predict,
        llm_rag_top_k=args.llm_rag_top_k,
        llm_timeout_connect=args.llm_timeout_connect,
        llm_timeout_read=args.llm_timeout_read,
        route_auto_accept_threshold=args.route_auto_accept_threshold,
        route_llm_review_threshold=args.route_llm_review_threshold,
        split_path=args.split_path,
        save_split_path=args.save_split_path,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_compare(cfg)
