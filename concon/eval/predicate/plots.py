import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import math
import re
import negation

# ------ complexity - memory interactions ------

class ComplexityMemory:
    # Runs differ by hidden size (48, 96, 192).
    # All have depth 1 - 2.
    # There are varying "n" property kinds with negation and no_negation.
    # There are varying "n-n" property kinds with conjunction.
    # Each run is repeated 5 times.
    # All other hyperparameters are equal.
    def __init__(self, datapoints, name=None):
        self.datapoints = datapoints
        self.name = name or "unnamed"
        self.hidden_colors = self._hidden_palette()

    def _hidden_palette(self):
        hidden_sizes = sorted({
            dp.get("config", {}).get("hidden_size")
            for dp in self.datapoints
            if dp.get("config", {}).get("hidden_size") is not None
        })
        if not hidden_sizes:
            return {}
        colors = sns.color_palette("viridis", n_colors=len(hidden_sizes))
        return {h: colors[i] for i, h in enumerate(hidden_sizes)}

    def _properties_sort_key(self, prop):
        s = str(prop)
        nums = tuple(int(x) for x in re.findall(r"\d+", s))
        if nums:
            return (0, nums, s)
        return (1, tuple(), s)

    def _build_epochs_df(self):
        '''
        datapoints (dict[str, list[dict]]) of runs with keys:
            - config (dict), 
            - evaluation (DataFrame), 
            - predicates (DataFrame),
            - languages (list[dict[epoch_number: int, DataFrame]])
        '''
        results = []
        missing = 0

        # Iterate through datapoints
        for dp in self.datapoints:
            eval_df = dp.get("evaluation")
            if eval_df is None or "eval/accuracy" not in eval_df.columns:
                missing += 1
                continue

            complexity = dp["config"].get("num_predicates")
            properties = dp["config"].get("properties")
            hidden     = dp["config"].get("hidden_size")
            # Find max accuracy/perf and the first epoch where those maxima are achieved.
            max_acc = eval_df["eval/accuracy"].max()
            best_epoch_acc = eval_df.loc[eval_df["eval/accuracy"] >= (max_acc - 0.001), "epoch"].min() + 1 # epochs are 0-index
            max_perf = eval_df["eval/perf"].max() if "eval/perf" in eval_df.columns else None
            best_epoch_perf = eval_df.loc[eval_df["eval/perf"] >= (max_perf - 0.001), "epoch"].min() + 1

            results.append({
                    'complexity': int(complexity),
                    'properties': properties,
                    'hidden_size': hidden,
                    'negation': "-neg" if dp["config"].get("no_negation", False) else "+neg",
                    'conjunction': "-conj" if dp["config"].get("no_conjunction", False) else "+conj",
                    'epochs_to_max_acc': best_epoch_acc,
                    'epochs_to_max_perf': best_epoch_perf,
                })

        if missing:
            print(f"[WARN] Missing at least {missing} evaluation data.")
        if not results:
            print(f"No valid evaluation data found in {self.name}.")
            return None
        return pd.DataFrame(results)

    def _tick_labels(self, plot_df, label_mode="full"):
        xticks = sorted(pd.to_numeric(plot_df["complexity"], errors="coerce").dropna().unique())
        if label_mode == "complexity":
            xtick_labels = [str(int(c)) for c in xticks]
        else:
            xtick_labels = []
            for c in xticks:
                if "properties" not in plot_df.columns:
                    props_str = "?"
                else:
                    props = (
                        plot_df.loc[plot_df["complexity"] == c, "properties"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    props = sorted(props, key=self._properties_sort_key)
                    props_str = "/".join(props) if len(props) else "?"
                if label_mode == "properties":
                    xtick_labels.append(props_str)
                else:
                    xtick_labels.append(f"{int(c)}: {props_str}")
        return xticks, xtick_labels

    def _plot_epochs_generic(self, plot_df, *, hue=None, filename="", label_mode="full", out_dir="plots"):
        fig = plt.figure(figsize=(10, 6))
        if hue is None:
            for y_col, label in [("epochs_to_max_acc", "max accuracy"), ("epochs_to_max_perf", "max performance")]:
                if y_col == "epochs_to_max_perf" and not plot_df["epochs_to_max_perf"].notna().any():
                    continue
                stats = (
                    plot_df.groupby("complexity")[y_col]
                    .agg(mean="mean", lo="min", hi="max")
                    .reset_index()
                )
                plt.fill_between(stats["complexity"], stats["lo"], stats["hi"], alpha=0.15)
                plt.plot(stats["complexity"], stats["mean"], marker="o", label=label)
        else:
            # Combine accuracy/perf into a single long form so legend is not duplicated.
            long_df = plot_df.copy()
            long_df = long_df.melt(
                id_vars=[c for c in long_df.columns if c not in ["epochs_to_max_acc", "epochs_to_max_perf"]],
                value_vars=["epochs_to_max_acc", "epochs_to_max_perf"],
                var_name="metric",
                value_name="epochs_to_max",
            )
            long_df["metric"] = long_df["metric"].map({
                "epochs_to_max_acc": "max accuracy",
                "epochs_to_max_perf": "max performance",
            })
            # Drop perf rows if perf is missing.
            long_df = long_df[long_df["epochs_to_max"].notna()]
            hue_vals = list(long_df[hue].dropna().unique())
            if hue == "hidden_size":
                hue_vals = sorted(hue_vals)
                palette = sns.color_palette("viridis", n_colors=max(1, len(hue_vals)))
                hue_color = {h: self.hidden_colors.get(h, palette[i]) for i, h in enumerate(hue_vals)}
            elif hue == "negation":
                hue_color = {"+neg": "#1f77b4", "-neg": "#e07a2d"}
            else:
                palette = sns.color_palette("viridis", n_colors=max(1, len(hue_vals)))
                hue_color = {h: palette[i] for i, h in enumerate(hue_vals)}
            metric_style = {
                "max accuracy": ("-", "o"),
                "max performance": ("--", "s"),
            }
            for (hval, mval), grp in long_df.groupby([hue, "metric"]):
                stats = (
                    grp.groupby("complexity")["epochs_to_max"]
                    .agg(mean="mean", lo="min", hi="max")
                    .reset_index()
                )
                label = f"{hval} | {mval}"
                color = hue_color.get(hval)
                linestyle, marker = metric_style.get(mval, ("-", "o"))
                plt.fill_between(stats["complexity"], stats["lo"], stats["hi"], alpha=0.15, color=color)
                plt.plot(
                    stats["complexity"],
                    stats["mean"],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                )

        plt.title('epochs until convergence by game complexity', fontweight='bold')
        plt.xlabel('game complexity')
        plt.ylabel('epochs to max')
        plt.grid(True, alpha=0.3)
        plt.legend()
        xticks, xtick_labels = self._tick_labels(plot_df, label_mode=label_mode)
        plt.xscale("log", base=2)
        plt.xticks(xticks, xtick_labels, rotation=45, ha="right")

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)

    def _plot_negation_scores_generic(self, neg_df, *, mode="greedy", filename="", label_mode="complexity", out_dir="plots"):
        prefix = f"{mode}_"
        score_cols = {
            f"{prefix}score_mi": "mutual information",
            f"{prefix}score_xor": "exclusivity",
            f"{prefix}score_purity": "purity",
        }
        required = ["complexity", "properties"] + list(score_cols.keys())
        missing = [c for c in required if c not in neg_df.columns]
        if missing:
            print(f"[WARN] Negation scores missing columns: {missing}")
            return

        plot_df = neg_df[["complexity", "properties"] + list(score_cols.keys())].copy()
        long_df = plot_df.melt(
            id_vars=["complexity", "properties"],
            value_vars=list(score_cols.keys()),
            var_name="metric",
            value_name="score",
        )
        long_df["metric"] = long_df["metric"].map(score_cols)
        long_df = long_df[long_df["score"].notna()]

        fig = plt.figure(figsize=(10, 6))
        for metric, grp in long_df.groupby("metric"):
            stats = (
                grp.groupby("complexity")["score"]
                .agg(mean="mean", lo="min", hi="max")
                .reset_index()
            )
            plt.fill_between(stats["complexity"], stats["lo"], stats["hi"], alpha=0.15)
            plt.plot(stats["complexity"], stats["mean"], marker="o", label=metric)
        plt.title(f'negation scores by number of predicates ({mode})', fontweight='bold')
        plt.xlabel('number of predicates')
        plt.ylabel('score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        xticks, xtick_labels = self._tick_labels(plot_df, label_mode=label_mode)
        plt.xscale("log", base=2)
        plt.xticks(xticks, xtick_labels, rotation=45, ha="right")

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)

    def plot_epochs_to_max(self, plot_df, out_dir="plots"):
        self._plot_epochs_generic(
            plot_df,
            hue=None,
            filename=f"epoch_to_max_metrics_{self.name}.png",
            label_mode="full",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_hidden(self, plot_df, out_dir="plots"):
        self._plot_epochs_generic(
            plot_df,
            hue="hidden_size",
            filename=f"epoch_to_max_acc_by_hidden_{self.name}.png",
            label_mode="full",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_negation(self, plot_df, out_dir="plots"):
        self._plot_epochs_generic(
            plot_df,
            hue="negation",
            filename=f"epoch_to_max_acc_by_negation_{self.name}.png",
            label_mode="full",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_conjunction(self, plot_df, out_dir="plots"):
        self._plot_epochs_generic(
            plot_df,
            hue="conjunction",
            filename=f"epoch_to_max_acc_by_conjunction_{self.name}.png",
            label_mode="full",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_negation_and_conjunction(self, plot_df, out_dir="plots"):
        plot_df = plot_df.copy()
        plot_df["neg_conj"] = plot_df["negation"] + "/" + plot_df["conjunction"]
        self._plot_epochs_generic(
            plot_df,
            hue="neg_conj",
            filename=f"epoch_to_max_acc_by_neg_conj_{self.name}.png",
            label_mode="full",
            out_dir=out_dir,
        )

    def plot_time_to_max_accuracy(self, out_dir="plots"):
        plot_df = self._build_epochs_df()
        if plot_df is None:
            return
        self.plot_epochs_to_max(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_hidden(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_negation(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_conjunction(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_negation_and_conjunction(plot_df, out_dir=out_dir)
        summary = plot_df.groupby('complexity')[['epochs_to_max_acc', 'epochs_to_max_perf']].agg(['mean', 'std', 'count'])
        print(summary)

    def plot_negation_scores(self, neg_df=None, *, mode="greedy", out_dir="plots", label_mode="complexity"):
        if not hasattr(negation, "normalize_negation_df"):
            print("[WARN] plot_negation_scores uses legacy schema and is disabled for current negation outputs.")
            return
        if neg_df is None:
            out_dir_path = pathlib.Path(__file__).resolve().parent / "outputs"
            csv_path = out_dir_path / f"negation_analysis_{self.name}.csv"
            if not csv_path.is_file():
                print(f"[WARN] Negation analysis not found: {csv_path}")
                return
            neg_df = pd.read_csv(csv_path)

        neg_df = negation.normalize_negation_df(neg_df, mode)
        filename = f"negation_scores_{mode}_{self.name}.png"
        self._plot_negation_scores_generic(
            neg_df,
            mode=mode,
            filename=filename,
            label_mode=label_mode,
            out_dir=out_dir,
        )

    def summarize_negation_by_complexity(self, neg_df=None, *, mode="greedy", threshold=0.9, out_dir="outputs"):
        if not hasattr(negation, "normalize_negation_df"):
            print("[WARN] summarize_negation_by_complexity uses legacy schema and is disabled for current negation outputs.")
            return None
        if neg_df is None:
            out_dir_path = pathlib.Path(__file__).resolve().parent / "outputs"
            csv_path = out_dir_path / f"negation_analysis_{self.name}.csv"
            if not csv_path.is_file():
                print(f"[WARN] Negation analysis not found: {csv_path}")
                return None
            neg_df = pd.read_csv(csv_path)

        neg_df = negation.normalize_negation_df(neg_df, mode)
        prefix = f"{mode}_"
        score_cols = [f"{prefix}score_mi", f"{prefix}score_xor", f"{prefix}score_purity"]
        required = ["complexity", "properties"] + score_cols
        missing = [c for c in required if c not in neg_df.columns]
        if missing:
            print(f"[WARN] Negation summary missing columns: {missing}")
            return None

        df = neg_df[["complexity", "properties"] + score_cols].copy()
        df = df.rename(columns={
            f"{prefix}score_mi": "score_mi",
            f"{prefix}score_xor": "score_xor",
            f"{prefix}score_purity": "score_purity",
        })

        scores = df[["score_mi", "score_xor", "score_purity"]].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            hmean = np.where(
                np.all(scores > 0, axis=1),
                3.0 / (1.0 / scores).sum(axis=1),
                0.0,
            )
        df["score_hmean"] = hmean

        df["meets_mi"] = (df["score_mi"] >= threshold).astype(int)
        df["meets_xor"] = (df["score_xor"] >= threshold).astype(int)
        df["meets_purity"] = (df["score_purity"] >= threshold).astype(int)
        df["meets_all"] = df["meets_mi"] & df["meets_xor"] & df["meets_purity"]

        summary = df.groupby("properties").agg(
            n_runs=("score_mi", "size"),
            avg_mi=("score_mi", "mean"),
            avg_exclusivity=("score_xor", "mean"),
            avg_purity=("score_purity", "mean"),
            avg_hmean=("score_hmean", "mean"),
            frac_mi_ge_thr=("meets_mi", "mean"),
            frac_excl_ge_thr=("meets_xor", "mean"),
            frac_purity_ge_thr=("meets_purity", "mean"),
            frac_all_ge_thr=("meets_all", "mean"),
        ).reset_index()

        out_dir_path = pathlib.Path(__file__).resolve().parent / out_dir
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = out_dir_path / f"negation_summary_{mode}_{self.name}.csv"
        summary.to_csv(out_path, index=False)
        print(f"Saved negation summary to: {out_path}")
        return summary

    def plot_message_compression(self, *, group_by=("negation",), out_dir="plots"):
        """
        Plot normalized message length over epochs.
        Normalization: msg_length / log_A(M), where A=2N+1 and M=N (no neg) or 2N (with neg).
        """
        groups = {}
        epochs = None

        for i, dp in enumerate(self.datapoints):
            eval_df = dp.get("evaluation")
            if eval_df is None or "eval/msg_length" not in eval_df.columns:
                continue
            if epochs is None:
                epochs = eval_df["epoch"].to_numpy()

            cfg = dp.get("config", {})
            N = int(cfg.get("num_predicates"))
            has_neg = not cfg.get("no_negation", False)
            A = int(cfg.get("base_alphabet_size"))
            theoretical_min = math.log(N, A) if N > 1 else 1.0
            normed = eval_df["eval/msg_length"].to_numpy(dtype=float) / theoretical_min

            key = tuple(cfg.get(k) if k not in ("negation", "conjunction") else None for k in group_by)
            if "negation" in group_by:
                key = tuple(("+neg" if has_neg else "-neg") if k == "negation" else v for k, v in zip(group_by, key))
            if "conjunction" in group_by:
                has_conj = not cfg.get("no_conjunction", False)
                key = tuple(("+conj" if has_conj else "-conj") if k == "conjunction" else v for k, v in zip(group_by, key))

            groups.setdefault(key, []).append(normed)

        if not groups or epochs is None:
            print(f"[WARN] No message length data found for {self.name}.")
            return

        def _sort_key(key):
            norm = []
            for v in key:
                if v is None:
                    norm.append((1, ""))
                else:
                    try:
                        norm.append((0, int(v)))
                    except (TypeError, ValueError):
                        if isinstance(v, str):
                            norm.append((0, self._properties_sort_key(v)))
                        else:
                            norm.append((0, str(v)))
            return tuple(norm)

        ordered_keys = sorted(groups.keys(), key=_sort_key)
        if tuple(group_by) == ("hidden_size",):
            palette = sns.color_palette("viridis", n_colors=max(1, len(ordered_keys)))
            color_map = {k: self.hidden_colors.get(k[0], palette[i]) for i, k in enumerate(ordered_keys)}
        elif tuple(group_by) == ("negation",):
            color_map = {("+neg",): "#1f77b4", ("-neg",): "#e07a2d"}
            fallback_palette = sns.color_palette("viridis", n_colors=max(1, len(ordered_keys)))
            for i, k in enumerate(ordered_keys):
                if k not in color_map:
                    color_map[k] = fallback_palette[i]
        else:
            palette = sns.color_palette("viridis", n_colors=max(1, len(ordered_keys)))
            color_map = {k: palette[i] for i, k in enumerate(ordered_keys)}

        fig = plt.figure(figsize=(10, 6))
        for key in ordered_keys:
            curves = groups[key]
            curves = np.array(curves)
            mean_y = curves.mean(axis=0)
            median_y = np.median(curves, axis=0)
            min_y = curves.min(axis=0)
            max_y = curves.max(axis=0)

            label = ", ".join(f"{k}={v}" for k, v in zip(group_by, key))
            color = color_map.get(key)
            plt.plot(epochs, mean_y, label=f"{label} (mean)", alpha=0.9, linewidth=2, color=color)
            plt.plot(epochs, median_y, color=color, linestyle=":", alpha=0.7, label=f"{label} (median)")
            plt.fill_between(epochs, min_y, max_y, color=color, alpha=0.15)

            z = np.polyfit(epochs, mean_y, 1)
            trend = np.poly1d(z)
            plt.plot(epochs, trend(epochs), linestyle="--", color=color, alpha=0.5)

        plt.axhline(y=1.0, color="red", linestyle="-", alpha=0.3, label="perfect compression (1.0)")
        plt.xlabel("epochs")
        plt.ylabel("inverse compression ratio (actual / theoretical minimum)")
        plt.title("normalized message efficiency over time", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        groupby_name = "_".join(group_by)
        out_path = out_dir / f"message_compression_by_{groupby_name}_{self.name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)

    def _extract_run_id(self, run_name):
        m = re.search(r"__run=(\d+)", str(run_name))
        return int(m.group(1)) if m else np.nan

    def _build_run_meta_df(self):
        rows = []
        for dp in self.datapoints:
            cfg = dp.get("config", {})
            rows.append({
                "run_name": str(dp.get("run_name", "")),
                "complexity": cfg.get("num_predicates"),
                "hidden_size": cfg.get("hidden_size"),
                "properties": str(cfg.get("properties")),
                "no_negation": cfg.get("no_negation", False),
                "no_conjunction": cfg.get("no_conjunction", False),
            })
        return pd.DataFrame(rows).drop_duplicates(subset=["run_name"])

    def _build_topsim_df(self):
        rows = []
        topsim_col_candidates = [
            "eval/topsim_intensional_levenshtein",
            "topsim_intensional_levenshtein",
        ]
        for dp in self.datapoints:
            eval_df = dp.get("evaluation")
            if eval_df is None or len(eval_df) == 0:
                continue
            topsim_col = None
            for c in topsim_col_candidates:
                if c in eval_df.columns:
                    topsim_col = c
                    break
            if topsim_col is None:
                continue
            if "eval/perf" in eval_df.columns:
                row = eval_df.loc[eval_df["eval/perf"].idxmax()]
            elif "eval/accuracy" in eval_df.columns:
                row = eval_df.loc[eval_df["eval/accuracy"].idxmax()]
            else:
                row = eval_df.iloc[-1]
            rows.append({
                "run_name": str(dp.get("run_name", "")),
                "topsim_intensional_levenshtein": float(row[topsim_col]),
            })
        return pd.DataFrame(rows) if rows else None

    def _prepare_negation_top_df(self, neg_df):
        if neg_df is None or len(neg_df) == 0:
            return None
        req = {"run_name", "n", "l_X", "l_T", "r", "atoms", "support"}
        if not req.issubset(set(neg_df.columns)):
            print(f"[WARN] Negation dataframe missing columns: {sorted(req - set(neg_df.columns))}")
            return None

        df = neg_df.copy()
        df["run_name"] = df["run_name"].astype(str)
        meta = self._build_run_meta_df()[["run_name", "complexity", "hidden_size", "no_negation", "no_conjunction", "properties"]]
        needed_meta = [c for c in ["complexity", "hidden_size", "no_negation", "no_conjunction"] if c not in df.columns]
        if needed_meta:
            df = df.merge(meta[["run_name"] + needed_meta], on="run_name", how="left")
        if "properties" not in df.columns:
            df = df.merge(meta[["run_name", "properties"]], on="run_name", how="left")
        if "no_negation" in df.columns:
            df = df[df["no_negation"] == False]
        for c in ["complexity", "hidden_size", "n", "l_X", "l_T", "r", "atoms", "support"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "properties" in df.columns:
            df["properties"] = df["properties"].astype(str)
        df["orth_x"] = 1.0 - df["l_X"]
        df["orth_t"] = 1.0 - df["l_T"]
        return df.dropna(subset=["complexity", "n"])

    def _complexity_ticks(self, df):
        xticks = sorted(pd.to_numeric(df["complexity"], errors="coerce").dropna().unique())
        labels = []
        for c in xticks:
            if "properties" not in df.columns:
                labels.append(str(int(c)))
                continue
            props = df.loc[df["complexity"] == c, "properties"].dropna().astype(str).unique()
            props = sorted(props, key=self._properties_sort_key)
            if not props:
                labels.append(str(int(c)))
            else:
                labels.append(f"{int(c)} ({'/'.join(props)})")
        return xticks, labels

    def plot_negation_metrics_by_complexity(self, neg_df, *, out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metric_cols = [("n", "n"), ("orth_x", "1-l_X"), ("r", "r")]
        metric_colors = {"n": "#1f77b4", "orth_x": "#2ca02c", "r": "#d62728"}

        fig = plt.figure(figsize=(10, 6))
        for col, label in metric_cols:
            stats = df.groupby("complexity")[col].agg(mean="mean", median="median", lo="min", hi="max").reset_index()
            c = metric_colors[col]
            plt.fill_between(stats["complexity"], stats["lo"], stats["hi"], alpha=0.15, color=c)
            plt.plot(stats["complexity"], stats["mean"], marker="o", color=c, label=f"{label} mean")
            plt.plot(stats["complexity"], stats["median"], marker="o", linestyle=":", color=c, label=f"{label} median")
        xticks, labels = self._complexity_ticks(df)
        plt.xscale("log", base=2)
        plt.xticks(xticks, labels, rotation=45, ha="right")
        plt.title("negation metrics by complexity")
        plt.xlabel("complexity")
        plt.ylabel("score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"negation_metrics_by_complexity_{self.name}.png", dpi=150)

        if df["hidden_size"].notna().any():
            hidden_vals = sorted(df["hidden_size"].dropna().unique())
            fallback = sns.color_palette("viridis", n_colors=max(1, len(hidden_vals)))
            hcols = {h: self.hidden_colors.get(h, fallback[i]) for i, h in enumerate(hidden_vals)}
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
            for ax, (col, label) in zip(axes, metric_cols):
                for h, grp in df.groupby("hidden_size"):
                    stats = grp.groupby("complexity")[col].agg(mean="mean", lo="min", hi="max").reset_index()
                    c = hcols.get(h)
                    ax.fill_between(stats["complexity"], stats["lo"], stats["hi"], alpha=0.12, color=c)
                    ax.plot(stats["complexity"], stats["mean"], marker="o", color=c, label=f"h{h}")
                ax.set_title(label)
                ax.grid(True, alpha=0.3)
            for ax in axes:
                xticks, labels = self._complexity_ticks(df)
                ax.set_xscale("log", base=2)
                ax.set_xticks(xticks)
                ax.set_xticklabels(labels, rotation=45, ha="right")
            axes[0].set_ylabel("score")
            axes[1].set_xlabel("complexity")
            handles, labels_ = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=min(4, max(1, len(labels_))), frameon=False)
            fig.suptitle("negation metrics by complexity and hidden size")
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            fig.savefig(out_dir / f"negation_metrics_by_complexity_by_hidden_{self.name}.png", dpi=150)

    def plot_negation_metric_slopes(self, neg_df, *, profile_tag="", out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        df["run_id"] = df["run_name"].map(self._extract_run_id)
        slope_df = df.groupby(["run_id", "complexity"]).agg(
            n=("n", "mean"),
            orth_x=("orth_x", "mean"),
            r=("r", "mean"),
        ).reset_index()

        metric_cols = [("n", "n"), ("orth_x", "1-l_X"), ("r", "r")]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
        for ax, (col, label) in zip(axes, metric_cols):
            for rid, grp in slope_df.groupby("run_id"):
                grp = grp.sort_values("complexity")
                if len(grp) < 2:
                    continue
                ax.plot(grp["complexity"], grp[col], color="#808080", alpha=0.35, linewidth=1.0)
            agg = slope_df.groupby("complexity")[col].agg(mean="mean", median="median").reset_index()
            ax.plot(agg["complexity"], agg["mean"], color="#1f77b4", marker="o", linewidth=2.2, label="mean")
            ax.plot(agg["complexity"], agg["median"], color="#1f77b4", marker="o", linestyle=":", linewidth=2.0, label="median")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            xticks, labels = self._complexity_ticks(slope_df)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("score")
        axes[1].set_xlabel("complexity")
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False)
        tag = str(profile_tag) if profile_tag else "default"
        fig.suptitle(f"negation metric slopes across complexity ({tag})")
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.savefig(out_dir / f"negation_metric_slopes_{tag}_{self.name}.png", dpi=150)

    def plot_topsim_vs_negation(self, neg_df, *, profile_tag="", out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        topsim_df = self._build_topsim_df()
        if df is None or df.empty or topsim_df is None:
            return
        merged = df.merge(topsim_df, on="run_name", how="inner").dropna(subset=["topsim_intensional_levenshtein"])
        if merged.empty:
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metric_cols = [("n", "n"), ("orth_x", "1-l_X"), ("r", "r")]
        cvals = np.log2(np.maximum(merged["complexity"].astype(float), 1.0))
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
        scatter_obj = None
        for ax, (col, label) in zip(axes, metric_cols):
            scatter_obj = ax.scatter(
                merged["topsim_intensional_levenshtein"],
                merged[col],
                c=cvals,
                cmap="viridis",
                alpha=0.85,
                edgecolors="none",
            )
            x = merged["topsim_intensional_levenshtein"].to_numpy(dtype=float)
            y = merged[col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2 and np.ptp(x[mask]) > 1e-12:
                a, b = np.polyfit(x[mask], y[mask], deg=1)
                xg = np.linspace(x[mask].min(), x[mask].max(), 100)
                ax.plot(xg, a * xg + b, color="#111111", linewidth=2.0)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("topsim intensional levenshtein")
        axes[0].set_ylabel("negation metric")
        if scatter_obj is not None:
            cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
            cbar = fig.colorbar(scatter_obj, cax=cax)
            cbar.set_label("log2(complexity)")
        tag = str(profile_tag) if profile_tag else "default"
        fig.suptitle(f"topsim vs negation structure ({tag})")
        fig.subplots_adjust(left=0.06, right=0.89, bottom=0.14, top=0.86, wspace=0.30)
        fig.savefig(out_dir / f"topsim_vs_negation_{tag}_{self.name}.png", dpi=150)

    def plot_topsim_interaction_n(self, neg_df, *, profile_tag="", out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        topsim_df = self._build_topsim_df()
        if df is None or df.empty or topsim_df is None:
            return
        merged = df.merge(topsim_df, on="run_name", how="inner").dropna(subset=["topsim_intensional_levenshtein"])
        if merged.empty:
            return

        x = merged["topsim_intensional_levenshtein"].to_numpy(dtype=float)
        y = merged["n"].to_numpy(dtype=float)
        z = np.log2(np.maximum(merged["complexity"].to_numpy(dtype=float), 1.0))
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if mask.sum() < 4:
            return
        x, y, z = x[mask], y[mask], z[mask]
        X = np.column_stack([np.ones_like(x), x, z, x * z])
        b0, b1, b2, b3 = np.linalg.lstsq(X, y, rcond=None)[0]
        xg = np.linspace(x.min(), x.max(), 100)
        z_lo, z_hi = np.quantile(z, [0.25, 0.75])
        y_lo = b0 + b1 * xg + b2 * z_lo + b3 * xg * z_lo
        y_hi = b0 + b1 * xg + b2 * z_hi + b3 * xg * z_hi

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8, 6))
        sc = plt.scatter(x, y, c=z, cmap="viridis", alpha=0.80, edgecolors="none")
        plt.plot(xg, y_lo, color="#1f77b4", linewidth=2.2, label="low complexity (q25)")
        plt.plot(xg, y_hi, color="#d62728", linewidth=2.2, label="high complexity (q75)")
        plt.xlabel("topsim intensional levenshtein")
        plt.ylabel("n")
        tag = str(profile_tag) if profile_tag else "default"
        plt.title(f"topsim x complexity interaction for n ({tag})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        cbar = plt.colorbar(sc)
        cbar.set_label("log2(complexity)")
        fig.tight_layout()
        fig.savefig(out_dir / f"topsim_n_interaction_{tag}_{self.name}.png", dpi=150)
