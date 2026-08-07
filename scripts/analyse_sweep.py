#!/usr/bin/env python3
"""
Analyse a Bayesian (or random) hyperparameter sweep to locate *regions* that
reliably perform well, rather than the single lucky top run.

What it produces
----------------
1. Top-k summary       : per-hyperparameter distribution among the best runs
                         (median + IQR). Narrow IQR => the metric cares about
                         this param and prefers that band; wide IQR => freedom.
2. Importance          : permutation importance of a random-forest surrogate
                         (more honest than impurity-based importance).
3. Partial dependence  : "all else equal" shape of the metric vs each param,
                         computed from the surrogate so it is NOT biased by how
                         densely the sampler visited each region. Log-scale
                         params are handled in log space.
4. Pairwise surface    : 2D partial dependence for the two most important
                         numeric params, to reveal broad basins vs sharp ridges
                         and interactions.

Caveat this script exists to address: in a Bayesian sweep the sampler
deliberately over-explores promising regions, so raw "mean metric at each value"
plots are biased toward wherever the optimiser spent its budget. The surrogate +
partial dependence marginalises the other axes properly and sidesteps that.

Usage
-----
    python analyse_sweep.py \
        --entity myteam --project signalling \
        --sweep abcd1234 \
        --metric val/topsim --goal maximize \
        --top-frac 0.15 \
        --log-params lr entropy_coef \
        --out ./sweep_report

You can also skip W&B and analyse an exported CSV:
    python analyse_sweep.py --from-csv runs.csv --metric val/topsim --goal maximize ...
The CSV must have one row per run: metric column + one column per hyperparameter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_from_wandb(entity: str, project: str, sweep: str, metric: str) -> pd.DataFrame:
    """Pull every finished run in a sweep into a flat DataFrame (config + metric)."""
    import wandb  # imported lazily so the CSV path has no wandb dependency

    api = wandb.Api()
    sweep_obj = api.sweep(f"{entity}/{project}/{sweep}")
    rows = []
    for run in sweep_obj.runs:
        if run.state != "finished":
            continue
        # config keys prefixed with '_' are wandb internals; skip them
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        val = run.summary.get(metric, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        cfg[metric] = val
        cfg["_run_id"] = run.id
        rows.append(cfg)

    if not rows:
        sys.exit(f"No finished runs with metric '{metric}' found in sweep {sweep}.")
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} finished runs from sweep {sweep}.")
    return df


def load_from_csv(path: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if metric not in df.columns:
        sys.exit(f"Metric column '{metric}' not in CSV. Columns: {list(df.columns)}")
    print(f"Loaded {len(df)} runs from {path}.")
    return df


# --------------------------------------------------------------------------- #
# Column classification
# --------------------------------------------------------------------------- #
def classify_params(
    df: pd.DataFrame, metric: str, log_params: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """
    Split hyperparameter columns into (numeric, categorical, dropped).
    Dropped = constant across the sweep (carries no information) or bookkeeping.
    """
    ignore = {metric, "_run_id"}
    numeric, categorical, dropped = [], [], []
    for col in df.columns:
        if col in ignore:
            continue
        series = df[col]
        if series.nunique(dropna=True) <= 1:
            dropped.append(col)
            continue
        # try to coerce to numeric; object columns with mixed types -> categorical
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() > 0.99:  # essentially all numeric
            df[col] = coerced
            numeric.append(col)
        else:
            categorical.append(col)

    unknown_logs = set(log_params) - set(numeric)
    if unknown_logs:
        print(f"  (note: --log-params {sorted(unknown_logs)} are not numeric/varying; ignored)")
    return numeric, categorical, dropped


# --------------------------------------------------------------------------- #
# 1. Top-k summary
# --------------------------------------------------------------------------- #
def top_k_summary(
    df: pd.DataFrame, metric: str, goal: str, top_frac: float,
    numeric: list[str], categorical: list[str],
) -> pd.DataFrame:
    ascending = goal == "minimize"
    k = max(3, int(round(len(df) * top_frac)))
    best = df.sort_values(metric, ascending=ascending).head(k)
    print(f"\n=== Top-{k} runs ({top_frac:.0%}) — where do the winners live? ===")
    print(f"    metric {metric} in top set: "
          f"[{best[metric].min():.4g}, {best[metric].max():.4g}], "
          f"median {best[metric].median():.4g}")

    records = []
    for col in numeric:
        vals = best[col].dropna()
        full = df[col].dropna()
        q25, q75 = vals.quantile([0.25, 0.75])
        # spread ratio: how tight is the top-set band vs the whole search range?
        full_range = full.max() - full.min()
        tightness = (q75 - q25) / full_range if full_range else np.nan
        records.append({
            "param": col, "kind": "num",
            "top_median": vals.median(),
            "top_IQR": f"[{q25:.4g}, {q75:.4g}]",
            "rel_width": round(tightness, 2),  # ~0 => sharp preference, ~1 => insensitive
        })
    for col in categorical:
        counts = best[col].value_counts(normalize=True)
        top_val = counts.index[0]
        records.append({
            "param": col, "kind": "cat",
            "top_median": f"{top_val} ({counts.iloc[0]:.0%})",
            "top_IQR": "", "rel_width": "",
        })
    summary = pd.DataFrame(records)
    print("\n    rel_width ~0 => winners cluster tightly (param matters, use that band)")
    print("    rel_width ~1 => winners spread out (metric insensitive, you have freedom)\n")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(summary.to_string(index=False))
    return summary


# --------------------------------------------------------------------------- #
# Surrogate model (shared by importance + partial dependence)
# --------------------------------------------------------------------------- #
def build_design_matrix(
    df: pd.DataFrame, numeric: list[str], categorical: list[str], log_params: list[str],
):
    """Return X (encoded), feature names, and per-feature metadata for inverse transforms."""
    from sklearn.preprocessing import OrdinalEncoder

    frames, names, meta = [], [], []
    for col in numeric:
        vals = df[col].to_numpy(dtype=float)
        if col in log_params:
            if np.any(vals <= 0):
                print(f"  (warning: {col} has non-positive values; not log-transforming)")
                col_vals, is_log = vals, False
            else:
                col_vals, is_log = np.log10(vals), True
        else:
            col_vals, is_log = vals, False
        frames.append(col_vals.reshape(-1, 1))
        names.append(col)
        meta.append({"name": col, "kind": "num", "log": is_log})

    if categorical:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_arr = enc.fit_transform(df[categorical].astype(str))
        for i, col in enumerate(categorical):
            frames.append(cat_arr[:, i].reshape(-1, 1))
            names.append(col)
            meta.append({"name": col, "kind": "cat", "categories": list(enc.categories_[i])})

    X = np.hstack(frames) if frames else np.empty((len(df), 0))
    return X, names, meta


def fit_surrogate(X, y):
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features=0.7,
        n_jobs=-1, random_state=0,
    )
    rf.fit(X, y)
    return rf


# --------------------------------------------------------------------------- #
# 2. Permutation importance
# --------------------------------------------------------------------------- #
def importance(rf, X, y, names) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    r = permutation_importance(rf, X, y, n_repeats=30, random_state=0, n_jobs=-1)
    imp = (
        pd.DataFrame({"param": names, "importance": r.importances_mean, "std": r.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    oob = getattr(rf, "oob_score_", None)
    r2 = rf.score(X, y)
    print("\n=== Parameter importance (permutation, surrogate RF) ===")
    print(f"    surrogate in-sample R^2 = {r2:.3f}"
          + (f", oob = {oob:.3f}" if oob is not None else "")
          + "  (low R^2 => metric is mostly noise/seed, trust regions not points)\n")
    with pd.option_context("display.width", 120):
        print(imp.to_string(index=False))
    return imp


# --------------------------------------------------------------------------- #
# 3 & 4. Partial dependence plots
# --------------------------------------------------------------------------- #
def _axis_ticks(m, grid):
    """Return (tick_positions, tick_labels, xlabel) for a feature grid, honouring log."""
    if m["kind"] == "num" and m.get("log"):
        lo, hi = grid.min(), grid.max()
        # nice decade ticks in log10 space, labelled in natural units
        decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
        decades = decades[(decades >= lo) & (decades <= hi)]
        return decades, [f"{10**t:.2g}" for t in decades], f"{m['name']} (log scale)"
    if m["kind"] == "cat":
        cats = m["categories"]
        return np.arange(len(cats)), cats, m["name"]
    return None, None, m["name"]


def partial_dependence_plots(rf, X, names, meta, imp, metric, goal, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import partial_dependence

    order = imp["param"].tolist()
    idx = {n: i for i, n in enumerate(names)}
    meta_by = {m["name"]: m for m in meta}

    # ---- 1D partial dependence for each param, ranked by importance ----
    n = len(order)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), order):
        m = meta_by[name]
        pd_res = partial_dependence(rf, X, [idx[name]], kind="average", grid_resolution=60)
        grid = pd_res["grid_values"][0]
        avg = pd_res["average"][0]
        ax.plot(grid, avg, marker="." if m["kind"] == "cat" else None)
        ticks, labels, xlabel = _axis_ticks(m, grid)
        if ticks is not None:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=0 if m["kind"] == "num" else 30)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"PD: {metric}")
        ax.set_title(f"importance rank {order.index(name) + 1}", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Partial dependence of {metric} (goal: {goal})", y=1.02)
    fig.tight_layout()
    p1 = out / "partial_dependence_1d.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved 1D partial dependence -> {p1}")

    # ---- 2D surface for the two most important NUMERIC params ----
    num_ranked = [nm for nm in order if meta_by[nm]["kind"] == "num"]
    if len(num_ranked) >= 2:
        a, b = num_ranked[0], num_ranked[1]
        pd_res = partial_dependence(rf, X, [(idx[a], idx[b])], kind="average", grid_resolution=40)
        ga, gb = pd_res["grid_values"]
        Z = pd_res["average"][0]  # shape (len(ga), len(gb))
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cf = ax.contourf(ga, gb, Z.T, levels=14, cmap="viridis")
        fig.colorbar(cf, ax=ax, label=f"PD: {metric}")
        for nm, setter, grid in ((a, ax.set_xticks, ga), (b, ax.set_yticks, gb)):
            ticks, labels, lbl = _axis_ticks(meta_by[nm], grid)
            if ticks is not None:
                setter(ticks)
                (ax.set_xticklabels if setter == ax.set_xticks else ax.set_yticklabels)(labels)
            (ax.set_xlabel if nm == a else ax.set_ylabel)(lbl)
        ax.set_title(f"{metric}: {a} x {b}\nbroad basin => robust, thin ridge => fragile")
        p2 = out / "partial_dependence_2d.png"
        fig.savefig(p2, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 2D partial dependence ({a} x {b}) -> {p2}")
        print("    broad coloured basin => robust sweet spot; thin ridge => fragile optimum")
    else:
        print("Skipped 2D surface (need >=2 varying numeric params).")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sweep", help="W&B sweep id (requires --entity/--project)")
    src.add_argument("--from-csv", help="analyse an exported CSV instead of hitting W&B")
    ap.add_argument("--entity")
    ap.add_argument("--project")
    ap.add_argument("--metric", required=True, help="summary metric column to optimise")
    ap.add_argument("--goal", choices=["maximize", "minimize"], default="maximize")
    ap.add_argument("--top-frac", type=float, default=0.15, help="fraction of runs = 'winners'")
    ap.add_argument("--log-params", nargs="*", default=[], help="params to treat in log10 space")
    ap.add_argument("--out", default="./sweep_report", help="output dir for plots + csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        df = load_from_csv(args.from_csv, args.metric)
    else:
        if not (args.entity and args.project):
            sys.exit("--entity and --project are required with --sweep.")
        df = load_from_wandb(args.entity, args.project, args.sweep, args.metric)

    df.to_csv(out / "runs.csv", index=False)

    numeric, categorical, dropped = classify_params(df, args.metric, args.log_params)
    print(f"\nNumeric params:    {numeric}")
    print(f"Categorical params: {categorical}")
    if dropped:
        print(f"Dropped (constant): {dropped}")
    if not numeric and not categorical:
        sys.exit("No varying hyperparameters found — nothing to analyse.")

    summary = top_k_summary(df, args.metric, args.goal, args.top_frac, numeric, categorical)
    summary.to_csv(out / "top_k_summary.csv", index=False)

    y = df[args.metric].to_numpy(dtype=float)
    # RF regresses the raw metric; for 'minimize' the shapes are just read inverted,
    # but we keep sign so PD y-axis matches the real metric.
    X, names, meta = build_design_matrix(df, numeric, categorical, args.log_params)
    rf = fit_surrogate(X, y)
    imp = importance(rf, X, y, names)
    imp.to_csv(out / "importance.csv", index=False)

    partial_dependence_plots(rf, X, names, meta, imp, args.metric, args.goal, out)

    print(f"\nDone. Report written to {out.resolve()}/")
    print("Reminder: confirm any candidate region by re-running its configs across "
          "several seeds before trusting it.")


if __name__ == "__main__":
    main()
