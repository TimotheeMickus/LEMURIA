#!/usr/bin/env python3
"""
Analyze a W&B grid search.

Which hyperparameters vary is detected automatically: any run-config key that
takes more than one value across the runs (excluding per-run identifiers) is
treated as a swept hyperparameter.

Produces:
  (i)  One summary plot per hyperparameter (min / mean+/-std / median / max).
  (ii) One run-time plot per hyperparameter (mean and median run time).
  (iii) One figure per pair of hyperparameters: mean and best (max/min) grids.
  (iv) A top-10 ranking of hyperparameter combinations.

Output is either a single combined PDF (default) or independent PNG figures,
selected with --format.

--metric selects the metric analyzed (default eval/perf). --direction says
whether higher or lower is better. --filter x drops the worst fraction x of
runs (ranked by final eval/perf) before any analysis.

Usage:
    python analyze_gridsearch.py
    python analyze_gridsearch.py --project entity/GridSearch-July --outdir figs
    python analyze_gridsearch.py --metric eval/loss --direction minimize
    python analyze_gridsearch.py --filter 0.5 --format png
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

# Reserved (non-config) columns of the assembled DataFrame.
RESERVED = {"state", "perf", "filter_perf", "runtime"}

# Config keys never treated as swept hyperparameters, even if they vary.
IGNORE_PARAMS = {"pretrain_epochs", "summary", "no_spigot"}


def _scalar(v):
    """True for values we can group/pivot on directly."""
    return isinstance(v, (int, float, str, bool)) or v is None


def _best_over_history(run, metric, maximize):
    """Best value of `metric` over the whole run history (max if maximizing,
    min if minimizing). Uses scan_history for the exact extreme. Returns None
    if the metric was never logged."""
    try:
        vals = [row.get(metric) for row in run.scan_history(keys=[metric])]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return None
        return max(vals) if maximize else min(vals)
    except Exception:
        return None


def fetch_runs(project, metric, point, maximize):
    """One row per finished run: every scalar config key + analysis metric
    ('perf') + filtering metric ('filter_perf', always final eval/perf) +
    runtime. `perf` is the final value (from summary) if point=='final', or the
    best value over the run history if point=='best'."""
    api = wandb.Api()
    runs = api.runs(project)
    if point == "best":
        print("point='best': reading run histories (slower)...", flush=True)

    records = []
    for i, run in enumerate(runs):
        print(f"fetching run {i}...", flush=True)
        if point == "best":
            perf = _best_over_history(run, metric, maximize)
        else:
            perf = run.summary.get(metric)
        row = {
            "state": run.state,
            "perf": perf,
            "filter_perf": run.summary.get(FILTER_METRIC),  # always final
            "runtime": run.summary.get("_runtime"),  # run duration in seconds
        }
        for k, v in run.config.items():
            # Keep scalars as-is; stringify anything else so it stays groupable.
            row[k] = v if _scalar(v) else str(v)
        records.append(row)

    df = pd.DataFrame(records)
    # Need the filtering metric present to rank/filter; keep only finished runs.
    df = df[(df["state"] == "finished") & df["filter_perf"].notna()].copy()
    return df


def detect_params(df):
    """Config keys that vary across runs, excluding per-run identifiers.

    A swept grid axis takes several values but each repeats across runs, so it
    has 1 < nunique < n_runs. Identifiers like 'name' are unique per run
    (nunique == n_runs) and are excluded."""
    n = len(df)
    params = []
    for c in df.columns:
        if c in RESERVED or c in IGNORE_PARAMS or c.startswith("_"):
            continue
        k = df[c].nunique(dropna=False)
        if 1 < k < n:
            params.append(c)
    return sorted(params)


def apply_filter(df, x, maximize):
    """Keep the best (1 - x) fraction of runs, ranked by filter_perf.
    'best' = highest filter_perf when maximizing, lowest when minimizing."""
    if x <= 0:
        return df
    n_keep = max(1, round(len(df) * (1.0 - x)))
    kept = (df.nlargest(n_keep, "filter_perf") if maximize
            else df.nsmallest(n_keep, "filter_perf")).copy()
    end = "highest" if maximize else "lowest"
    print(f"filter={x}: keeping best {len(kept)}/{len(df)} runs "
          f"({end} {FILTER_METRIC})", flush=True)
    return kept


def sorted_values(df, param):
    """Unique values of a param, sorted numerically when possible."""
    vals = df[param].dropna().unique().tolist()
    try:
        return sorted(vals, key=float)
    except (TypeError, ValueError):
        return sorted(vals, key=str)


def rank_top(df, maximize, n=10):
    """Return the top-n hyperparameter combinations, best first."""
    return (df.dropna(subset=["perf"])
              .sort_values("perf", ascending=not maximize)
              .head(n))


def print_top(top, metric, params, maximize, point):
    """Print the top ranking to stdout."""
    sense = "highest" if maximize else "lowest"
    print(f"\nTop {len(top)} combinations by {point} {metric} ({sense} first):")
    header = "  rank  " + "  ".join(f"{p:>18}" for p in params) + f"  {'perf':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        vals = "  ".join(f"{str(r[p]):>18}" for p in params)
        print(f"  {rank:>4}  {vals}  {r['perf']:>10.4f}")
    print()


def build_summary_figure(config, top, metric, params, maximize, point):
    """A text page: the run configuration and the top-10 ranking."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Grid search analysis", fontsize=16, fontweight="bold", y=0.97)

    lines = ["Configuration", "-------------"]
    width = max(len(name) for name, _, _ in config)
    for name, value, desc in config:
        lines.append(f"{name:<{width}} = {value!r:<20}  ({desc})")

    lines += ["", f"Detected varying hyperparameters: {', '.join(params)}"]

    sense = "highest" if maximize else "lowest"
    lines += ["", f"Top {len(top)} combinations by {point} {metric} "
              f"({sense} first)", "-" * 40]
    header = "rank  " + "  ".join(f"{p:>16}" for p in params) + f"  {'perf':>9}"
    lines.append(header)
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        vals = "  ".join(f"{str(r[p]):>16}" for p in params)
        lines.append(f"{rank:>4}  {vals}  {r['perf']:>9.4f}")

    fig.text(0.05, 0.90, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=8)
    return fig


def build_single_figures(df, metric, params, point):
    """(i) One summary figure per hyperparameter: min, max (whiskers),
    mean +/- std (band), mean and median markers. Returns (name, fig) list."""
    figures = []
    for p in params:
        stats = df.groupby(p)["perf"].agg(["min", "max", "mean", "median", "std"])
        counts = df.groupby(p)["perf"].count()
        vals = sorted_values(df, p)
        stats = stats.reindex(vals)
        counts = counts.reindex(vals).fillna(0).astype(int)
        std = stats["std"].fillna(0.0)

        x = np.arange(len(vals))
        half = 0.28

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
        ax.set_ylabel(f"{point} {metric}")
        ax.set_title(f"{metric} by {p}  (min / mean+/-std / median / max)")
        ax.grid(axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        figures.append((f"single_{p}", fig))
    return figures


def build_runtime_figures(df, params):
    """One grouped bar chart per hyperparameter: average and median run time
    (in minutes) for each value. Returns a list of (name, figure)."""
    figures = []
    if "runtime" not in df.columns or df["runtime"].notna().sum() == 0:
        print("no runtime data available; skipping run-time figures")
        return figures

    rt = df["runtime"] / 60.0  # seconds -> minutes
    for p in params:
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


def build_pair_figures(df, metric, params, maximize):
    """(iii) One figure per pair of hyperparameters, with two grids side by
    side: mean metric and best metric (max when maximizing, min when
    minimizing). Returns (name, figure) list."""
    figures = []
    best_agg = "max" if maximize else "min"
    best_label = best_agg  # "max" or "min"
    for p1, p2 in itertools.combinations(params, 2):
        rows_o, cols_o = sorted_values(df, p1), sorted_values(df, p2)

        def grid(aggfunc):
            g = df.pivot_table(values="perf", index=p1, columns=p2,
                               aggfunc=aggfunc)
            return g.reindex(index=rows_o, columns=cols_o)

        mean_g = grid("mean")
        best_g = grid(best_agg)
        cnt = grid("count")

        fig, (ax_mean, ax_best) = plt.subplots(1, 2, figsize=(12, 5))
        _draw_heatmap(fig, ax_mean, mean_g, cnt, p1, p2,
                      f"Mean {metric}: {p1} vs {p2}", f"mean {metric}")
        _draw_heatmap(fig, ax_best, best_g, cnt, p1, p2,
                      f"{best_label.capitalize()} {metric}: {p1} vs {p2}",
                      f"{best_label} {metric}")
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
    parser.add_argument("--direction", choices=["maximize", "minimize"],
                        default="maximize",
                        help="Whether higher (maximize) or lower (minimize) "
                             "values of --metric are better. Default: maximize.")
    parser.add_argument("--point", choices=["final", "best"], default="final",
                        help="Take --metric at the end of each run ('final', "
                             "from summary) or its best over training ('best', "
                             "max/min per --direction, read from history and "
                             "slower). Default: final.")
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

    maximize = args.direction == "maximize"

    # Configuration summary (printed and, for pdf, added as a page).
    config = [
        ("--project", args.project,
         "any W&B project string, e.g. 'entity/project'"),
        ("--outdir", args.outdir,
         "any writable directory path (created if missing)"),
        ("--metric", args.metric,
         "any logged metric key, e.g. 'eval/perf'"),
        ("--direction", args.direction,
         "'maximize' or 'minimize' (which is better for --metric)"),
        ("--point", args.point,
         "'final' (end of run) or 'best' (max/min over training)"),
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
              f"_dir-{args.direction}"
              f"_point-{args.point}"
              f"_filter-{args.filter_x:g}")

    df = fetch_runs(args.project, args.metric, args.point, maximize)
    print(f"\n{len(df)} usable runs")

    params = detect_params(df)
    if not params:
        raise SystemExit("No varying hyperparameters detected across runs.")
    print(f"detected varying hyperparameters: {', '.join(params)}")

    df = apply_filter(df, args.filter_x, maximize)
    print(f"{len(df)} runs after filtering\n")

    top = rank_top(df, maximize, n=10)
    print_top(top, args.metric, params, maximize, args.point)

    figures = build_single_figures(df, args.metric, params, args.point) \
        + build_runtime_figures(df, params) \
        + build_pair_figures(df, args.metric, params, maximize)

    if args.format == "pdf":
        summary = build_summary_figure(config, top, args.metric, params,
                                       maximize, args.point)
        figures = [("summary", summary)] + figures
        emit_pdf(figures, args.outdir, suffix)
    else:
        emit_png(figures, args.outdir, suffix)

    print(f"\nDone. Output in ./{args.outdir}/")


if __name__ == "__main__":
    main()
