#!/usr/bin/env python3
"""Merge reviewer feedback into the labeled gold-standard CSV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def read_csv_flex(path: Path) -> pd.DataFrame:
    last_exc: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_exc


def normalize_label(x: Any) -> str:
    s = str(x or "").strip().lower()
    if "primary" in s:
        return "Primary"
    if "reuse" in s:
        return "Reuse"
    return "Unclear"


def row_key(row: pd.Series) -> Tuple[str, str]:
    paper_id = str(row.get("paper_id", "") or row.get("PaperID", "")).strip()
    identifier = str(row.get("identifier", "")).strip()
    title = str(row.get("title", "")).strip().lower()
    if paper_id:
        return ("paper_id", paper_id)
    if identifier:
        return ("identifier", identifier)
    return ("title", title)


def merge_feedback(base_csv: Path, feedback_csv: Path, output_csv: Path) -> Dict[str, Any]:
    base_df = read_csv_flex(base_csv)
    if not feedback_csv.exists():
        output_csv.write_text(base_df.to_csv(index=False), encoding="utf-8-sig")
        return {
            "output_csv": str(output_csv),
            "feedback_exists": False,
            "updated_rows": 0,
            "appended_rows": 0,
            "merged_rows": int(len(base_df)),
        }

    feedback_df = read_csv_flex(feedback_csv)
    if feedback_df.empty:
        output_csv.write_text(base_df.to_csv(index=False), encoding="utf-8-sig")
        return {
            "output_csv": str(output_csv),
            "feedback_exists": True,
            "updated_rows": 0,
            "appended_rows": 0,
            "merged_rows": int(len(base_df)),
        }

    label_col = "ground_truth" if "ground_truth" in base_df.columns else "human_label"
    if label_col not in base_df.columns:
        raise ValueError("Base CSV must contain ground_truth or human_label.")

    for col in [
        "feedback_corrected_label",
        "feedback_predicted_label",
        "feedback_decision",
        "feedback_reviewer",
        "feedback_note",
        "feedback_timestamp",
        "feedback_route",
        "feedback_route_reason",
    ]:
        if col not in base_df.columns:
            base_df[col] = ""

    key_to_index = {row_key(row): idx for idx, row in base_df.iterrows()}
    appended = 0
    updated = 0

    for _, fb in feedback_df.iterrows():
        key = row_key(fb)
        corrected = normalize_label(fb.get("corrected_label", ""))
        if corrected == "Unclear":
            continue
        timestamp = str(fb.get("timestamp_utc", "")).strip() or datetime.now(timezone.utc).isoformat()
        if key in key_to_index:
            idx = key_to_index[key]
            base_df.at[idx, label_col] = corrected
            base_df.at[idx, "feedback_corrected_label"] = corrected
            base_df.at[idx, "feedback_predicted_label"] = str(fb.get("predicted_label", ""))
            base_df.at[idx, "feedback_decision"] = str(fb.get("feedback_decision", ""))
            base_df.at[idx, "feedback_reviewer"] = str(fb.get("reviewer", ""))
            base_df.at[idx, "feedback_note"] = str(fb.get("note", ""))
            base_df.at[idx, "feedback_timestamp"] = timestamp
            base_df.at[idx, "feedback_route"] = str(fb.get("recommended_route", ""))
            base_df.at[idx, "feedback_route_reason"] = str(fb.get("recommended_route_reason", ""))
            updated += 1
        else:
            row = {col: "" for col in base_df.columns}
            row["paper_id"] = str(fb.get("paper_id", ""))
            if "title" in row:
                row["title"] = str(fb.get("title", ""))
            if "article_url" in row:
                row["article_url"] = ""
            if "gse" in row:
                row["gse"] = str(fb.get("gse_ids", ""))
            row[label_col] = corrected
            row["feedback_corrected_label"] = corrected
            row["feedback_predicted_label"] = str(fb.get("predicted_label", ""))
            row["feedback_decision"] = str(fb.get("feedback_decision", ""))
            row["feedback_reviewer"] = str(fb.get("reviewer", ""))
            row["feedback_note"] = str(fb.get("note", ""))
            row["feedback_timestamp"] = timestamp
            row["feedback_route"] = str(fb.get("recommended_route", ""))
            row["feedback_route_reason"] = str(fb.get("recommended_route_reason", ""))
            base_df = pd.concat([base_df, pd.DataFrame([row])], ignore_index=True)
            key_to_index[row_key(pd.Series(row))] = len(base_df) - 1
            appended += 1

    base_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return {
        "output_csv": str(output_csv),
        "feedback_exists": True,
        "updated_rows": updated,
        "appended_rows": appended,
        "merged_rows": int(len(base_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge feedback CSV into a curated gold-standard CSV.")
    parser.add_argument("--base-csv", type=Path, default=Path("manual_ground_truth_with_GSE_links_REFRESHED.csv"))
    parser.add_argument("--feedback-csv", type=Path, default=Path("rag_feedback_gold_standard.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("manual_ground_truth_with_feedback_merged.csv"))
    args = parser.parse_args()

    summary = merge_feedback(args.base_csv, args.feedback_csv, args.output_csv)
    if not summary["feedback_exists"]:
        print(f"[info] Feedback file not found. Copied base CSV to {args.output_csv}")
        return
    print(f"[ok] Wrote {summary['output_csv']}")
    print(f"[ok] Updated rows: {summary['updated_rows']}")
    print(f"[ok] Appended rows: {summary['appended_rows']}")


if __name__ == "__main__":
    main()
