#!/usr/bin/env python3
"""
Analyze a W&B grid search.

Produces:
  (i)  One summary plot per hyperparameter (min / mean+/-std / median / max).
  (ii) One heatmap per pair of hyperparameters (average metric per cell).
  (iii) A top-10 ranking of hyperparameter combinations.

Output is either a single combined PDF (default) or independent PNG figures,
selected with --format.

The metric analyzed is configurable (--metric, default eval/perf).
Runs can be filtered to the best fraction (--filter x): the worst x of runs,
ranked by final eval/perf (always eval/perf, regardless of --metric), are
dropped before any analysis.

Usage:
    python analyze_gridsearch.py
    python analyze_gridsearch.py --project entity/GridSearch-July --outdir figs
    python analyze_gridsearch.py --metric eval/accuracy --filter 0.5
    python analyze_gridsearch.py --format png
"""

import argparse
import itertools
import os
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import wandb

# We intentionally build every figure before emitting (the PDF path writes them
# all into one file), so more than 20 are open at once. They are all closed at
# emit time, so silence matplotlib's precautionary "too many open figures" warning.
plt.rcParams["figure.max_open_warning"] = 0

# Metric always used for filtering the worst runs, independent of --metric.
FILTER_METRIC = "eval/perf"

PARAMS = [
    "hidden_size",
    "num_candidates",
    "predicate_sampling",
    "learning_rate",
    "beta_asker",
]


def fetch_runs(project, metric):
    """One row per finished run: params + analysis metric ('perf') +
    filtering metric ('filter_perf', always eval/perf)."""
    api = wandb.Api()
    runs = api.runs(project)

    records = []
    for i, run in enumerate(runs):
        print(f"fetching run {i}...", flush=True)
        row = {
            "state": run.state,
            "perf": run.summary.get(metric),
            "filter_perf": run.summary.get(FILTER_METRIC),
            "runtime": run.summary.get("_runtime"),  # run duration in seconds
        }
        for p in PARAMS:
            row[p] = run.config.get(p)
        records.append(row)

    df = pd.DataFrame(records)
    # Need the filtering metric present to rank/filter; keep only finished runs.
    df = df[(df["state"] == "finished") & df["filter_perf"].notna()].copy()

    # Ensure numeric params sort numerically, not lexicographically.
    for p in PARAMS:
        if p != "predicate_sampling":
            df[p] = pd.to_numeric(df[p], errors="coerce")

    return df


def apply_filter(df, x):
    """Keep the best (1 - x) fraction of runs, ranked by filter_perf."""
    if x <= 0:
        return df
    n_keep = max(1, round(len(df) * (1.0 - x)))
    kept = df.nlargest(n_keep, "filter_perf").copy()
    print(f"filter={x}: keeping best {len(kept)}/{len(df)} runs "
          f"by {FILTER_METRIC}", flush=True)
    return kept


def sorted_values(df, param):
    """Unique values of a param, sorted numerically when possible."""
    vals = df[param].dropna().unique().tolist()
    try:
        return sorted(vals, key=float)
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def rank_top(df, n=10):
    """Return the top-n hyperparameter combinations ranked by the metric."""
    return (df.dropna(subset=["perf"])
              .sort_values("perf", ascending=False)
              .head(n))


def print_top(top, metric):
    """Print the top ranking to stdout."""
    print(f"\nTop {len(top)} combinations by {metric}:")
    header = "  rank  " + "  ".join(f"{p:>18}" for p in PARAMS) + f"  {'perf':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        vals = "  ".join(f"{str(r[p]):>18}" for p in PARAMS)
        print(f"  {rank:>4}  {vals}  {r['perf']:>10.4f}")
    print()


def build_summary_figure(config, top, metric):
    """A text page: the run configuration and the top-10 ranking."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Grid search analysis", fontsize=16, fontweight="bold", y=0.97)

    lines = ["Configuration", "-------------"]
    width = max(len(name) for name, _, _ in config)
    for name, value, desc in config:
        lines.append(f"{name:<{width}} = {value!r:<20}  ({desc})")

    lines += ["", f"Top {len(top)} combinations by {metric}", "-" * 40]
    header = "rank  " + "  ".join(f"{p:>16}" for p in PARAMS) + f"  {'perf':>9}"
    lines.append(header)
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        vals = "  ".join(f"{str(r[p]):>16}" for p in PARAMS)
        lines.append(f"{rank:>4}  {vals}  {r['perf']:>9.4f}")

    fig.text(0.05, 0.90, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=8)
    return fig


def build_single_figures(df, metric):
    """(i) Build one summary figure per hyperparameter.
    Returns a list of (name, figure)."""
    figures = []
    for p in PARAMS:
        stats = df.groupby(p)["perf"].agg(["min", "max", "mean", "median", "std"])
        counts = df.groupby(p)["perf"].count()  # runs with a valid metric value
        vals = sorted_values(df, p)
        stats = stats.reindex(vals)
        counts = counts.reindex(vals).fillna(0).astype(int)
        std = stats["std"].fillna(0.0)  # std is NaN when a group has one run

        x = np.arange(len(vals))
        half = 0.28  # half-width of the std box

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for xi, (_, row), s in zip(x, stats.iterrows(), std.values):
            if np.isnan(row["mean"]):
                continue
            ax.plot([xi, xi], [row["min"], row["max"]], color="#888888",
                    lw=1.2, zorder=1)
            ax.plot([xi - 0.08, xi + 0.08], [row["min"], row["min"]],
                    color="#888888", lw=1.2, zorder=1)
            ax.plot([xi - 0.08, xi + 0.08], [row["max"], row["max"]],
                    color="#888888", lw=1.2, zorder=1)
            ax.add_patch(plt.Rectangle(
                (xi - half, row["mean"] - s), 2 * half, 2 * s,
                facecolor="#4C72B0", alpha=0.30, edgecolor="none", zorder=2))
            ax.plot([xi - half, xi + half], [row["mean"]] * 2,
                    color="#1F3B73", lw=2.0, zorder=3)
            ax.plot([xi - half, xi + half], [row["median"]] * 2,
                    color="#C44E52", lw=1.6, ls="--", zorder=3)

        handles = [
            Line2D([0], [0], color="#888888", lw=1.2, label="min / max"),
            Patch(facecolor="#4C72B0", alpha=0.30, label="mean +/- std"),
            Line2D([0], [0], color="#1F3B73", lw=2.0, label="mean"),
            Line2D([0], [0], color="#C44E52", lw=1.6, ls="--", label="median"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n(n={counts[v]})" for v in vals])
        ax.set_xlim(-0.5, len(vals) - 0.5)
        ax.set_xlabel(p)
        ax.set_ylabel(f"final {metric}")
        ax.set_title(f"{metric} by {p}  (min / mean+/-std / median / max)")
        ax.grid(axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        figures.append((f"single_{p}", fig))
    return figures


def build_runtime_figures(df):
    """One grouped bar chart per hyperparameter: average and median run time
    (in minutes) for each value. Returns a list of (name, figure)."""
    figures = []
    if "runtime" not in df.columns or df["runtime"].notna().sum() == 0:
        print("no runtime data available; skipping run-time figures")
        return figures

    rt = df["runtime"] / 60.0  # seconds -> minutes
    for p in PARAMS:
        g = df.assign(_rt=rt).groupby(p)["_rt"]
        stats = g.agg(["mean", "median"])
        counts = g.count()
        vals = sorted_values(df, p)
        stats = stats.reindex(vals)
        counts = counts.reindex(vals).fillna(0).astype(int)

        x = np.arange(len(vals))
        w = 0.38

        fig, ax = plt.subplots(figsize=(7, 4.5))
        b1 = ax.bar(x - w / 2, stats["mean"].values, w, label="mean",
                    color="#4C72B0")
        b2 = ax.bar(x + w / 2, stats["median"].values, w, label="median",
                    color="#C44E52")
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                if not np.isnan(h):
                    ax.text(rect.get_x() + rect.get_width() / 2, h,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n(n={counts[v]})" for v in vals])
        ax.set_xlim(-0.5, len(vals) - 0.5)
        ax.set_xlabel(p)
        ax.set_ylabel("run time (minutes)")
        ax.set_title(f"Run time by {p}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        figures.append((f"runtime_{p}", fig))
    return figures


def _draw_heatmap(fig, ax, values, cnt, p1, p2, title, cbar_label):
    """Draw a single labelled heatmap (values grid + per-cell n) onto ax."""
    im = ax.imshow(values.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels([str(c) for c in values.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(values.index)))
    ax.set_yticklabels([str(i) for i in values.index])
    ax.set_xlabel(p2)
    ax.set_ylabel(p1)
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.values[i, j]
            if not np.isnan(v):
                n = int(cnt.values[i, j]) if not np.isnan(cnt.values[i, j]) else 0
                ax.text(j, i, f"{v:.3f}\nn={n}", ha="center", va="center",
                        color="white", fontsize=8)

    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)


def build_pair_figures(df, metric):
    """(ii) Build one figure per pair of hyperparameters, with two grids
    side by side: mean metric and max metric. Returns (name, figure) list."""
    figures = []
    for p1, p2 in itertools.combinations(PARAMS, 2):
        rows_o, cols_o = sorted_values(df, p1), sorted_values(df, p2)

        def grid(aggfunc):
            g = df.pivot_table(values="perf", index=p1, columns=p2,
                               aggfunc=aggfunc)
            return g.reindex(index=rows_o, columns=cols_o)

        mean_g = grid("mean")
        max_g = grid("max")
        cnt = grid("count")

        fig, (ax_mean, ax_max) = plt.subplots(1, 2, figsize=(12, 5))
        _draw_heatmap(fig, ax_mean, mean_g, cnt, p1, p2,
                      f"Mean {metric}: {p1} vs {p2}", f"mean {metric}")
        _draw_heatmap(fig, ax_max, max_g, cnt, p1, p2,
                      f"Max {metric}: {p1} vs {p2}", f"max {metric}")
        fig.tight_layout()
        figures.append((f"pair_{p1}__{p2}", fig))
    return figures


def emit_png(figures, outdir, suffix):
    """Save each figure as an independent PNG."""
    for name, fig in figures:
        path = os.path.join(outdir, f"{name}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


def emit_pdf(figures, outdir, suffix):
    """Save all figures into a single PDF."""
    path = os.path.join(outdir, f"analysis{suffix}.pdf")
    with PdfPages(path) as pdf:
        for _, fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="GridSearch-July",
                        help='W&B project, e.g. "entity/GridSearch-July"')
    parser.add_argument("--outdir", default="gridsearch_figs")
    parser.add_argument("--metric", default="eval/perf",
                        help="Metric to analyze (default: eval/perf).")
    parser.add_argument("--filter", type=float, default=0.0, dest="filter_x",
                        help="Fraction of worst runs (by final eval/perf) to "
                             "drop, in [0, 1]. 0 keeps all, 0.5 keeps the best "
                             "half. Default: 0.")
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf",
                        help="'pdf' bundles everything into one PDF (default); "
                             "'png' writes independent figure files.")
    args = parser.parse_args()

    if not 0.0 <= args.filter_x <= 1.0:
        parser.error("--filter must be in [0, 1]")

    # Configuration summary (printed and, for pdf, added as a page).
    config = [
        ("--project", args.project,
         "any W&B project string, e.g. 'entity/project'"),
        ("--outdir", args.outdir,
         "any writable directory path (created if missing)"),
        ("--metric", args.metric,
         "any logged metric key, e.g. 'eval/perf'"),
        ("--filter", args.filter_x,
         f"float in [0, 1]; drops that fraction of worst runs by {FILTER_METRIC}"),
        ("--format", args.format,
         "'pdf' (one combined file) or 'png' (independent files)"),
    ]
    print("Configuration:")
    width = max(len(name) for name, _, _ in config)
    for name, value, desc in config:
        print(f"  {name:<{width}} = {value!r:<20} ({desc})")
    print()

    os.makedirs(args.outdir, exist_ok=True)

    # Filesystem-safe suffix so different project/metric/filter values
    # don't overwrite each other.
    def safe(s):
        return re.sub(r"[^0-9A-Za-z]+", "-", str(s)).strip("-")

    suffix = (f"_project-{safe(args.project)}"
              f"_metric-{safe(args.metric)}"
              f"_filter-{args.filter_x:g}")

    df = fetch_runs(args.project, args.metric)
    print(f"\n{len(df)} usable runs")

    df = apply_filter(df, args.filter_x)
    print(f"{len(df)} runs after filtering\n")

    top = rank_top(df, n=10)
    print_top(top, args.metric)

    figures = build_single_figures(df, args.metric) \
        + build_runtime_figures(df) \
        + build_pair_figures(df, args.metric)

    if args.format == "pdf":
        # Lead with a summary page, then all the plots, in one file.
        figures = [("summary", build_summary_figure(config, top, args.metric))] \
            + figures
        emit_pdf(figures, args.outdir, suffix)
    else:
        emit_png(figures, args.outdir, suffix)

    print(f"\nDone. Output in ./{args.outdir}/")


if __name__ == "__main__":
    main()
