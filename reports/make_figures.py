"""Generate the 3 required evaluation figures from results/full.csv.

Usage:
    python -m reports.make_figures
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS = Path("results/full.csv")
OUT_DIR = Path("reports/figures")
METRIC_COLS = ["completion", "tool_f1", "args", "numeric", "faithfulness", "alignment"]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    df["ARS"] = df[METRIC_COLS].mean(axis=1)
    return df


def fig1_ars_by_system_tier(df: pd.DataFrame) -> None:
    """Bar chart: ARS by system × tier."""
    pivot = df.groupby(["system", "tier"])["ARS"].mean().unstack("tier")
    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title("ARS by System × Tier")
    ax.set_xlabel("System")
    ax.set_ylabel("Agent Reliability Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Tier")
    plt.tight_layout()
    out = OUT_DIR / "ars_by_tier.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved %s", out)


def fig2_metric_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: metric × tier for the agent system."""
    agent_df = df[df["system"] == "agent"]
    pivot = agent_df.groupby("tier")[METRIC_COLS].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels(METRIC_COLS, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Agent Metric × Tier Heatmap")
    plt.colorbar(im, ax=ax, label="Score")
    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(METRIC_COLS)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "metric_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved %s", out)


def fig3_failure_pie(df: pd.DataFrame) -> None:
    """Pie chart: failure type distribution for the agent."""
    agent_df = df[df["system"] == "agent"].copy()
    # Classify each row into the dominant failure mode
    def classify(row: pd.Series) -> str:
        if row["completion"] < 0.5:
            return "Completion failure"
        if row["tool_f1"] < 0.5:
            return "Wrong tool selection"
        if row["args"] < 0.5:
            return "Bad arguments"
        if row["numeric"] < 0.5:
            return "Numeric error"
        if row["faithfulness"] < 0.5:
            return "Hallucination"
        if row["alignment"] < 0.5:
            return "Plan misalignment"
        return "No major failure"

    agent_df["failure"] = agent_df.apply(classify, axis=1)
    counts = agent_df["failure"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=140)
    ax.set_title("Agent Failure Type Distribution")
    plt.tight_layout()
    out = OUT_DIR / "failure_pie.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved %s", out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    logger.info("Loaded %d rows from %s", len(df), RESULTS)
    fig1_ars_by_system_tier(df)
    fig2_metric_heatmap(df)
    fig3_failure_pie(df)
    print("Figures written to reports/figures/")


if __name__ == "__main__":
    main()
