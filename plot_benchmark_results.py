#!/usr/bin/env python3
"""Create presentation-ready benchmark plots for the provenance classifier.

This script is intentionally read-only with respect to benchmark CSV inputs.
It converts existing evaluation outputs into PNG figures so model comparisons
and LLM-vs-RAG status can be shown graphically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _load_main_outputs(out_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "model": _read_csv(out_dir / "model_comparison.csv"),
        "cv": _read_csv(out_dir / "cv_model_comparison.csv"),
        "sensitivity": _read_csv(out_dir / "evidence_window_sensitivity.csv"),
    }


def _build_llm_run_summary(run_dir: Path, run_label: str) -> pd.DataFrame:
    metrics = _read_csv(run_dir / "model_comparison.csv")
    preds = _read_csv(run_dir / "test_predictions.csv")

    rows: List[Dict[str, object]] = []
    rag_vote_acc = float(metrics.loc[metrics["model"] == "rag_vote", "accuracy_all"].iloc[0])
    rag_llm_rows = metrics.loc[metrics["model"] == "rag_llm"]
    rag_llm_acc = float(rag_llm_rows["accuracy_all"].iloc[0]) if not rag_llm_rows.empty else np.nan

    llm_called_mask = preds.get("pred_rag_llm_called", pd.Series([0] * len(preds))).fillna(0).astype(int) == 1
    llm_called_n = int(llm_called_mask.sum())
    total_n = int(len(preds))
    llm_call_rate = float(llm_called_n / total_n) if total_n else 0.0

    rows.append(
        {
            "run_label": run_label,
            "scope": "all_rows",
            "model": "rag_vote",
            "accuracy_all": rag_vote_acc,
            "n": total_n,
            "llm_called_rows": llm_called_n,
            "llm_call_rate": llm_call_rate,
        }
    )
    rows.append(
        {
            "run_label": run_label,
            "scope": "all_rows",
            "model": "rag_llm",
            "accuracy_all": rag_llm_acc,
            "n": total_n,
            "llm_called_rows": llm_called_n,
            "llm_call_rate": llm_call_rate,
        }
    )

    if llm_called_n > 0 and {"label", "pred_rag_vote", "pred_rag_llm"}.issubset(preds.columns):
        called_df = preds.loc[llm_called_mask].copy()
        rows.append(
            {
                "run_label": run_label,
                "scope": "llm_called_only",
                "model": "rag_vote",
                "accuracy_all": float((called_df["label"] == called_df["pred_rag_vote"]).mean()),
                "n": llm_called_n,
                "llm_called_rows": llm_called_n,
                "llm_call_rate": llm_call_rate,
            }
        )
        rows.append(
            {
                "run_label": run_label,
                "scope": "llm_called_only",
                "model": "rag_llm",
                "accuracy_all": float((called_df["label"] == called_df["pred_rag_llm"]).mean()),
                "n": llm_called_n,
                "llm_called_rows": llm_called_n,
                "llm_call_rate": llm_call_rate,
            }
        )

    return pd.DataFrame(rows)


def _plot_single_split(ax: plt.Axes, model_df: pd.DataFrame) -> None:
    order = ["static_rules", "mined_template_rules", "linear_model", "rag_vote", "rules_plus_rag", "rag_llm"]
    work = model_df.copy()
    work["model"] = pd.Categorical(work["model"], categories=order, ordered=True)
    work = work.sort_values("model")
    ax.bar(work["model"], work["accuracy_all"], color=["#8da0cb", "#66c2a5", "#fc8d62", "#1b9e77", "#e78ac3", "#d95f02"][: len(work)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Held-Out Split")
    ax.tick_params(axis="x", rotation=30)
    for idx, value in enumerate(work["accuracy_all"]):
        ax.text(idx, float(value) + 0.02, f"{float(value):.3f}", ha="center", va="bottom", fontsize=9)


def _plot_cv(ax: plt.Axes, cv_df: pd.DataFrame) -> None:
    keep = cv_df[cv_df["model"].isin(["linear_model", "rag_vote"])].copy()
    keep["label"] = keep["extraction_mode"] + "\n" + keep["model"]
    ax.bar(
        keep["label"],
        keep["accuracy_all_mean"],
        yerr=keep["accuracy_all_std"].fillna(0.0),
        capsize=4,
        color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b"][: len(keep)],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("CV Mean Accuracy")
    ax.set_title("5-Fold Cross-Validation")
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(keep["accuracy_all_mean"]):
        ax.text(idx, float(value) + 0.02, f"{float(value):.3f}", ha="center", va="bottom", fontsize=9)


def _plot_sensitivity(ax: plt.Axes, sensitivity_df: pd.DataFrame) -> None:
    keep = sensitivity_df[sensitivity_df["model"].isin(["linear_model", "rag_vote"])].copy()
    keep["label"] = keep["evidence_variant"] + "\n" + keep["model"]
    ax.barh(keep["label"], keep["accuracy_all_mean"], color="#4c78a8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("CV Mean Accuracy")
    ax.set_title("Evidence Input Sensitivity")
    for idx, value in enumerate(keep["accuracy_all_mean"]):
        ax.text(float(value) + 0.01, idx, f"{float(value):.3f}", va="center", fontsize=9)


def create_main_figure(out_dir: Path) -> Path:
    data = _load_main_outputs(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _plot_single_split(axes[0], data["model"])
    _plot_cv(axes[1], data["cv"])
    _plot_sensitivity(axes[2], data["sensitivity"])
    fig.suptitle("Primary vs Reuse Benchmark Overview", fontsize=15)
    fig.tight_layout()
    out_path = out_dir / "benchmark_overview.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def create_llm_status_figure(main_out_dir: Path, llm_run_specs: List[str]) -> Path:
    rows: List[pd.DataFrame] = []
    baseline_metrics = _read_csv(main_out_dir / "model_comparison.csv")
    baseline_rag = float(baseline_metrics.loc[baseline_metrics["model"] == "rag_vote", "accuracy_all"].iloc[0])
    rows.append(
        pd.DataFrame(
            [
                {
                    "run_label": "baseline_no_llm",
                    "scope": "all_rows",
                    "model": "rag_vote",
                    "accuracy_all": baseline_rag,
                    "n": int(baseline_metrics["n"].iloc[0]),
                    "llm_called_rows": 0,
                    "llm_call_rate": 0.0,
                }
            ]
        )
    )

    for spec in llm_run_specs:
        if "=" in spec:
            label, path_text = spec.split("=", 1)
        else:
            path_text = spec
            label = Path(spec).name
        rows.append(_build_llm_run_summary(Path(path_text), label))

    llm_df = pd.concat(rows, ignore_index=True)
    llm_df.to_csv(main_out_dir / "rag_llm_status_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 6))

    run_order = list(dict.fromkeys(llm_df["run_label"]))
    x = np.arange(len(run_order))
    width = 0.18
    series = [
        ("all_rows", "rag_vote", "#4c78a8", "RAG vote (all rows)"),
        ("all_rows", "rag_llm", "#f58518", "RAG+LLM (all rows)"),
        ("llm_called_only", "rag_vote", "#72b7b2", "RAG vote (LLM-called rows)"),
        ("llm_called_only", "rag_llm", "#e45756", "RAG+LLM (LLM-called rows)"),
    ]

    for offset_idx, (scope, model, color, label) in enumerate(series):
        values = []
        for run_label in run_order:
            match = llm_df[(llm_df["run_label"] == run_label) & (llm_df["scope"] == scope) & (llm_df["model"] == model)]
            values.append(float(match["accuracy_all"].iloc[0]) if not match.empty else np.nan)
        positions = x + (offset_idx - 1.5) * width
        ax.bar(positions, values, width=width, color=color, label=label)
        for pos, value in zip(positions, values):
            if np.isnan(value):
                continue
            ax.text(pos, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    coverage_lines = []
    for run_label in run_order:
        match = llm_df[llm_df["run_label"] == run_label]
        if match.empty:
            coverage_lines.append(f"{run_label}: no data")
            continue
        llm_called_rows = int(match["llm_called_rows"].max())
        total_n = int(match["n"].max())
        llm_call_rate = float(match["llm_call_rate"].max())
        coverage_lines.append(f"{run_label}: {llm_called_rows}/{total_n} LLM-called rows ({llm_call_rate:.1%})")

    ax.set_xticks(x)
    ax.set_xticklabels(run_order, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("RAG vs RAG+LLM Status")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(1.02, 0.98, "\n".join(coverage_lines), transform=ax.transAxes, va="top", fontsize=9)

    fig.tight_layout()
    out_path = main_out_dir / "rag_llm_status.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create benchmark plots from evidence_modeling outputs")
    p.add_argument("--main-out-dir", type=Path, required=True, help="Directory with model_comparison.csv, cv_model_comparison.csv, and evidence_window_sensitivity.csv")
    p.add_argument(
        "--llm-run",
        action="append",
        default=[],
        help="Optional LLM run comparison in label=path form. Can be repeated.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    main_plot = create_main_figure(args.main_out_dir)
    print(f"Saved: {main_plot}")
    if args.llm_run:
        llm_plot = create_llm_status_figure(args.main_out_dir, args.llm_run)
        print(f"Saved: {llm_plot}")


if __name__ == "__main__":
    main()
