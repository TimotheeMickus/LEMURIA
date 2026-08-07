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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
_AGGS = ("min", "max", "mean", "last")


def parse_metric_spec(metric: str) -> tuple[str, str]:
    """
    Split 'eval/loss.min' -> ('eval/loss', 'min'). No recognised suffix -> agg 'last'.
    Mirrors W&B's UI naming, but the aggregation is computed by us from run history,
    because '.min' etc. are NOT real keys in run.summary unless explicitly defined.
    """
    base, _, suffix = metric.rpartition(".")
    if base and suffix in _AGGS:
        return base, suffix
    return metric, "last"


def _resolve_metric_value(run, base: str, agg: str):
    """
    Return the aggregated metric for one run.
    Order of attempts (cheap -> expensive):
      1. summary holds a nested dict {'min':..,'max':..} (define_metric with summaries)
      2. summary holds the literal 'base.agg' key (some flattened setups)
      3. agg == 'last' and summary[base] is a scalar
      4. scan full logged history and aggregate (exact; every step, no downsampling)
    """
    summ = run.summary
    v = summ.get(base, None)
    if isinstance(v, dict) and agg in v:
        return v[agg]
    literal = summ.get(f"{base}.{agg}", None)
    if literal is not None:
        return literal
    if agg == "last" and v is not None and not isinstance(v, dict):
        return v

    # Fall back to history. scan_history returns every logged row (history() samples
    # to ~500 points and could miss the true min), keyed to just this metric.
    vals = [
        row[base]
        for row in run.scan_history(keys=[base])
        if row.get(base) is not None
    ]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return {"min": arr.min, "max": arr.max, "mean": arr.mean, "last": lambda: arr[-1]}[agg]()


def _progress(iterable, total, desc):
    """Wrap an iterable in a tqdm bar if tqdm is available; otherwise pass through."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="run")
    except ImportError:
        print(f"{desc} (install tqdm for a progress bar)...")
        return iterable


def load_from_wandb(
    entity: str, project: str, sweep: str, metric: str, limit: int | None = None
) -> pd.DataFrame:
    """Pull every finished run in a sweep into a flat DataFrame (config + metric)."""
    import wandb  # imported lazily so the CSV path has no wandb dependency
    from itertools import islice

    base, agg = parse_metric_spec(metric)
    if agg != "last":
        print(f"Metric: {agg}({base}) computed from run history "
              f"(exposed as column '{metric}').")
    else:
        print(f"Metric: last value of '{base}' from run summary.")

    api = wandb.Api()
    sweep_obj = api.sweep(f"{entity}/{project}/{sweep}")

    runs = sweep_obj.runs
    try:
        total = len(runs)  # triggers a count query; known up front
    except Exception:
        total = None
    if total is not None:
        print(f"Sweep has {total} run(s).")

    if limit is not None:
        shown = min(total, limit) if total is not None else limit
        print(f"** DEBUG: loading only the first {shown} run(s) (--limit); "
              f"results are NOT representative of the full sweep. **")
        run_iter = islice(runs, limit)
        total = shown
    else:
        run_iter = runs

    rows, missing = [], 0
    for run in _progress(run_iter, total, f"loading {sweep}"):
        if run.state != "finished":
            continue
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        val = _resolve_metric_value(run, base, agg)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing += 1
            continue
        cfg[metric] = float(val)  # store under the full spec so --metric matches downstream
        cfg["_run_id"] = run.id
        rows.append(cfg)

    if not rows:
        sys.exit(
            f"No finished runs yielded metric '{metric}'. "
            f"Check the base key '{base}' is actually logged (not just a summary alias)."
        )
    if missing:
        print(f"  ({missing} finished runs skipped: metric missing/NaN)")
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
    df: pd.DataFrame, metric: str
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
    return numeric, categorical, dropped


# --------------------------------------------------------------------------- #
# Log-space detection + confirmation
# --------------------------------------------------------------------------- #
LOG_MIN_ORDERS = 1.7  # ~50x dynamic range: below this, treat as linear


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 3))


def detect_log_params(df: pd.DataFrame, numeric: list[str]) -> list[dict]:
    """
    Flag numeric params that look log-sampled: strictly positive, wide dynamic
    range, and more symmetric in log space than in linear space. Returns evidence
    rows (all positive wide-range params), with 'detected' = passes symmetry test.
    """
    out = []
    for col in numeric:
        v = df[col].dropna().to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 4 or np.any(v <= 0):
            continue
        orders = float(np.log10(v.max() / v.min()))
        if orders < LOG_MIN_ORDERS:
            continue
        raw_sk, log_sk = _skew(v), _skew(np.log10(v))
        detected = abs(log_sk) <= abs(raw_sk)  # log makes it (weakly) more symmetric
        out.append({
            "param": col, "orders": round(orders, 2),
            "raw_skew": round(raw_sk, 2), "log_skew": round(log_sk, 2),
            "detected": detected,
        })
    return out


def resolve_log_params(
    df: pd.DataFrame, numeric: list[str], explicit, assume_yes: bool,
) -> list[str]:
    """
    Decide which params to treat as log10.
      - explicit is not None (--log-params given): use it verbatim (validated).
      - else: auto-detect, then confirm (interactive) / auto-accept (batch or -y).
    """
    if explicit is not None:
        chosen, bad_missing, bad_sign = [], [], []
        for p in explicit:
            if p not in numeric:
                bad_missing.append(p)
            elif (df[p].dropna() <= 0).any():
                bad_sign.append(p)  # can't log a non-positive param
            else:
                chosen.append(p)
        if bad_missing:
            print(f"  (note: --log-params {bad_missing} not numeric/varying; ignored)")
        if bad_sign:
            print(f"  (note: --log-params {bad_sign} have non-positive values; NOT logged)")
        return chosen

    cand = detect_log_params(df, numeric)
    if not cand:
        print("\nLog-space detection: no candidates "
              "(no positive param spans >= %.1f orders of magnitude)." % LOG_MIN_ORDERS)
        return []

    default_sel = [c["param"] for c in cand if c["detected"]]
    print("\n=== Log-space detection (no --log-params given) ===")
    print("    positive params with wide dynamic range; 'detected' passes the "
          "log-symmetry test.")
    tbl = pd.DataFrame(cand)[["param", "orders", "raw_skew", "log_skew", "detected"]]
    with pd.option_context("display.width", 120):
        print(tbl.to_string(index=False))
    print(f"\n    proposed --log-params: {default_sel or '(none)'}")

    if assume_yes:
        print("    auto-accepted (--yes).")
        return default_sel
    if not sys.stdin.isatty():
        print("    non-interactive shell: auto-accepting the proposed set. "
              "Pass --log-params explicitly (or -y) to control this in batch jobs.")
        return default_sel

    prompt = ("\nTreat proposed params as log-scale? "
              "[Y]es / [n]one / type a space-separated subset to override: ")
    try:
        resp = input(prompt).strip()
    except EOFError:
        return default_sel
    if resp == "" or resp.lower() in ("y", "yes"):
        return default_sel
    if resp.lower() in ("n", "no", "none"):
        return []
    valid = set(numeric)
    chosen = [t for t in resp.split() if t in valid]
    unknown = [t for t in resp.split() if t not in valid]
    if unknown:
        print(f"    (ignored unknown/non-numeric: {unknown})")
    return chosen


# --------------------------------------------------------------------------- #
# Group-by resolution (stratified analysis)
# --------------------------------------------------------------------------- #
GROUP_MAX_CARD = 8   # suggest params with <= this many distinct values as candidates
GROUP_HARD_CARD = 25  # refuse to group by a param with more distinct values than this


def resolve_group_params(
    df: pd.DataFrame, numeric: list[str], categorical: list[str],
    explicit, assume_yes: bool,
) -> list[str]:
    """
    Decide which params to stratify the analysis by.
      - explicit is not None (--group-by given): use verbatim (validated).
      - else: prompt with a candidate list; DEFAULT IS NONE (no grouping).
    Candidates = categoricals + low-cardinality numerics. High-cardinality /
    continuous params are rejected (would create a group per run).
    """
    def _validate(names, source):
        ok = []
        for p in names:
            if p not in numeric and p not in categorical:
                print(f"    (note: group-by '{p}' not a varying param; ignored)")
                continue
            card = df[p].nunique(dropna=True)
            if card > GROUP_HARD_CARD:
                print(f"    (note: group-by '{p}' has {card} distinct values "
                      f"(> {GROUP_HARD_CARD}); too many groups — ignored)")
                continue
            ok.append(p)
        return ok

    if explicit is not None:
        return _validate(explicit, "explicit")

    candidates = list(categorical) + [
        n for n in numeric if df[n].nunique(dropna=True) <= GROUP_MAX_CARD
    ]
    if not candidates:
        print("\nGroup-by: no low-cardinality categorical candidates; analysing all runs together.")
        return []

    print("\n=== Stratified analysis (no --group-by given) ===")
    print("    run the analysis independently for each combination of these params.")
    rows = [{"param": c,
             "n_values": df[c].nunique(dropna=True),
             "values": ", ".join(map(str, sorted(df[c].dropna().unique())[:6]))
                       + (" ..." if df[c].nunique() > 6 else "")}
            for c in candidates]
    with pd.option_context("display.width", 120, "display.max_colwidth", 60):
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n    default: None (analyse all runs together)")

    if assume_yes or not sys.stdin.isatty():
        why = "--yes" if assume_yes else "non-interactive shell"
        print(f"    {why}: no grouping.")
        return []

    prompt = ("\nStratify analysis by which param(s)? "
              "[N]one (default) / type a space-separated subset: ")
    try:
        resp = input(prompt).strip()
    except EOFError:
        return []
    if resp == "" or resp.lower() in ("n", "no", "none"):
        return []
    valid = set(candidates)
    chosen = [t for t in resp.split() if t in valid]
    unknown = [t for t in resp.split() if t not in valid]
    if unknown:
        print(f"    (ignored — not a candidate: {unknown})")
    return _validate(chosen, "prompt")


def _group_label(cols: list[str], key) -> str:
    if not isinstance(key, tuple):
        key = (key,)
    return ",".join(f"{c}={v}" for c, v in zip(cols, key))


def _safe_dirname(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9=._,+-]+", "_", label)


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
        # bright = better regardless of goal (flip cmap when lower is better)
        cmap = "viridis" if goal == "maximize" else "viridis_r"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cf = ax.contourf(ga, gb, Z.T, levels=14, cmap=cmap)
        better = "higher" if goal == "maximize" else "lower"
        fig.colorbar(cf, ax=ax, label=f"PD: {metric}  (bright = better, {better})")
        for nm, setter, grid in ((a, ax.set_xticks, ga), (b, ax.set_yticks, gb)):
            ticks, labels, lbl = _axis_ticks(meta_by[nm], grid)
            if ticks is not None:
                setter(ticks)
                (ax.set_xticklabels if setter == ax.set_xticks else ax.set_yticklabels)(labels)
            (ax.set_xlabel if nm == a else ax.set_ylabel)(lbl)
        ax.set_title(f"{metric}: {a} x {b}\nbroad bright region => robust sweet spot, "
                     f"thin bright ridge => fragile")
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
    ap.add_argument("--metric", required=True,
                    help="metric to optimise. Accepts a W&B-style aggregation suffix: "
                         "'eval/loss.min', '.max', '.mean', or '.last' (default if no "
                         "suffix). Aggregations are computed from run history. "
                         "For a min-loss objective use --goal minimize.")
    ap.add_argument("--goal", choices=["maximize", "minimize"], default="maximize")
    ap.add_argument("--top-frac", type=float, default=0.15, help="fraction of runs = 'winners'")
    ap.add_argument("--log-params", nargs="*", default=None,
                    help="params to treat in log10 space. If omitted, the script "
                         "auto-detects candidates and asks for confirmation. Pass with "
                         "no names (just --log-params) to force NONE.")
    ap.add_argument("--group-by", nargs="*", default=None, metavar="PARAM",
                    help="stratify: run the analysis independently for each combination "
                         "of these (categorical / low-cardinality) params. If omitted, "
                         "you're prompted (default None). Pass with no names to force None.")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="skip the log-param and group-by prompts; accept detected log "
                         "params and no grouping (use in SLURM / non-interactive jobs)")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="DEBUG: only load the first N runs. Useful because pulling "
                         "history for hundreds of runs is slow. Results are partial — "
                         "don't use for real analysis.")
    ap.add_argument("--out", default="./sweep_report", help="output dir for plots + csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        df = load_from_csv(args.from_csv, args.metric)
        if args.limit is not None and len(df) > args.limit:
            print(f"** DEBUG: truncating to first {args.limit} of {len(df)} rows (--limit). **")
            df = df.head(args.limit).copy()
    else:
        if not (args.entity and args.project):
            sys.exit("--entity and --project are required with --sweep.")
        df = load_from_wandb(args.entity, args.project, args.sweep, args.metric, limit=args.limit)

    df.to_csv(out / "runs.csv", index=False)

    numeric, categorical, dropped = classify_params(df, args.metric)
    print(f"\nNumeric params:    {numeric}")
    print(f"Categorical params: {categorical}")
    if dropped:
        print(f"Dropped (constant): {dropped}")
    if not numeric and not categorical:
        sys.exit("No varying hyperparameters found — nothing to analyse.")

    group_params = resolve_group_params(df, numeric, categorical, args.group_by, args.yes)
    log_params = resolve_log_params(df, numeric, args.log_params, args.yes)
    print(f"\nLog-scale params in use: {log_params or '(none)'}")
    print(f"Stratify by:             {group_params or '(none — all runs together)'}")

    if not group_params:
        run_analysis(df, args.metric, args.goal, args.top_frac, log_params, out)
    else:
        groups = list(df.groupby(group_params, dropna=False, sort=True))
        print(f"\nStratified into {len(groups)} group(s) by {group_params}.")
        for key, sub in groups:
            label = _group_label(group_params, key)
            sub_out = out / _safe_dirname(label)
            sub_out.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 72)
            print(f"GROUP  {label}   (n = {len(sub)} runs)")
            print("=" * 72)
            run_analysis(sub.copy(), args.metric, args.goal, args.top_frac,
                         log_params, sub_out, group_cols=group_params)

    print(f"\nDone. Report written to {out.resolve()}/")
    print("Reminder: confirm any candidate region by re-running its configs across "
          "several seeds before trusting it.")


MIN_GROUP_ROWS = 12   # below this, skip the RF surrogate (importance/PD) — too few runs
MIN_TOPK_ROWS = 4     # below this, skip the group entirely


def run_analysis(df, metric, goal, top_frac, log_params, out: Path,
                 group_cols: list[str] | None = None):
    """Run top-k + surrogate importance + partial dependence on one (sub)set of runs."""
    n = len(df)
    if n < MIN_TOPK_ROWS:
        print(f"  Only {n} run(s) — too few to analyse; skipping this group.")
        return

    numeric, categorical, dropped = classify_params(df, metric)
    # params we stratified on are constant here and get dropped automatically
    if group_cols:
        numeric = [c for c in numeric if c not in group_cols]
        categorical = [c for c in categorical if c not in group_cols]
    if not numeric and not categorical:
        print("  No varying params left within this group; skipping.")
        return

    summary = top_k_summary(df, metric, goal, top_frac, numeric, categorical)
    summary.to_csv(out / "top_k_summary.csv", index=False)

    if n < MIN_GROUP_ROWS:
        print(f"  Only {n} runs (< {MIN_GROUP_ROWS}) — skipping surrogate "
              f"importance/partial-dependence for this group (would be unreliable).")
        return

    y = df[metric].to_numpy(dtype=float)
    # RF regresses the raw metric; for 'minimize' the shapes are just read inverted,
    # but we keep sign so PD y-axis matches the real metric.
    X, names, meta = build_design_matrix(df, numeric, categorical, log_params)
    if X.shape[1] == 0:
        print("  No usable features; skipping surrogate.")
        return
    rf = fit_surrogate(X, y)
    imp = importance(rf, X, y, names)
    imp.to_csv(out / "importance.csv", index=False)
    partial_dependence_plots(rf, X, names, meta, imp, metric, goal, out)


if __name__ == "__main__":
    main()
