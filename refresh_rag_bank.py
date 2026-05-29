#!/usr/bin/env python3
"""Create a refreshed human-reviewed RAG bank from base labels plus feedback."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from merge_feedback_into_gold import merge_feedback


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the labeled RAG bank using reviewed feedback.")
    parser.add_argument("--base-csv", type=Path, default=Path("manual_ground_truth_with_GSE_links_REFRESHED.csv"))
    parser.add_argument("--feedback-csv", type=Path, default=Path("rag_feedback_gold_standard.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("rag_bank_refreshed.csv"))
    parser.add_argument("--report-json", type=Path, default=Path("rag_bank_refresh_report.json"))
    args = parser.parse_args()

    summary: Dict[str, Any] = merge_feedback(args.base_csv, args.feedback_csv, args.output_csv)
    summary.update(
        {
            "base_csv": str(args.base_csv),
            "feedback_csv": str(args.feedback_csv),
            "refreshed_at_utc": utc_now_iso(),
            "workflow_note": "Only reviewer-confirmed rows from feedback are folded into the refreshed RAG bank.",
            "next_step_example": (
                "Use --labeled-csv-path "
                f"{args.output_csv.name} in evidence_modeling.py or production configuration to evaluate the refreshed bank."
            ),
        }
    )
    args.report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Refreshed RAG bank CSV: {args.output_csv}")
    print(f"[ok] Refresh report: {args.report_json}")
    print(f"[ok] Updated rows: {summary['updated_rows']}")
    print(f"[ok] Appended rows: {summary['appended_rows']}")


if __name__ == "__main__":
    main()
