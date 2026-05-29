#!/usr/bin/env python3
"""Export curator-approved Supabase feedback to a CSV usable by refresh_rag_bank.py.

Only rows with:
- review_status = approved
- approved_for_rag = true
are exported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from supabase_feedback import approved_supabase_row_to_feedback_csv_row
from supabase_feedback import fetch_approved_feedback_rows
from supabase_feedback import write_feedback_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export approved Supabase feedback for RAG-bank refresh.")
    parser.add_argument("--output-csv", type=Path, default=Path("rag_feedback_gold_standard.csv"))
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    raw_rows = fetch_approved_feedback_rows(limit=args.limit)
    csv_rows = [approved_supabase_row_to_feedback_csv_row(item) for item in raw_rows]
    summary = write_feedback_csv(csv_rows, args.output_csv)
    print(f"[ok] Exported approved feedback rows: {summary['rows']}")
    print(f"[ok] Output CSV: {summary['output_csv']}")
    print("[next] Run refresh_rag_bank.py with this CSV to build a refreshed RAG bank.")


if __name__ == "__main__":
    main()
