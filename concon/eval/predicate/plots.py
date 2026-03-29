import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import math

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
        xticks = sorted(plot_df['complexity'].unique())
        if label_mode == "complexity":
            xtick_labels = [str(c) for c in xticks]
        else:
            xtick_labels = []
            for c in xticks:
                props = plot_df.loc[plot_df["complexity"] == c, "properties"].dropna().astype(int).unique()
                props_str = props[0] if len(props) else "?"
                xtick_labels.append(f"{c}: {props_str}")
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
            palette = sns.color_palette(n_colors=len(hue_vals))
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

        plt.title('epochs until convergence by number of predicates', fontweight='bold')
        plt.xlabel('number of predicates')
        plt.ylabel('epochs to max')
        plt.grid(True, alpha=0.3)
        plt.legend()
        xticks = sorted(plot_df['complexity'].unique())
        xtick_labels = []
        for c in xticks:
            props = plot_df.loc[plot_df["complexity"] == c, "properties"].dropna().astype(int).unique()
            props_str = props[0] if len(props) else "?"
            xtick_labels.append(str(props_str))
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
            label_mode="complexity",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_conjunction(self, plot_df, out_dir="plots"):
        self._plot_epochs_generic(
            plot_df,
            hue="conjunction",
            filename=f"epoch_to_max_acc_by_conjunction_{self.name}.png",
            label_mode="complexity",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_negation_and_conjunction(self, plot_df, out_dir="plots"):
        plot_df = plot_df.copy()
        plot_df["neg_conj"] = plot_df["negation"] + "/" + plot_df["conjunction"]
        self._plot_epochs_generic(
            plot_df,
            hue="neg_conj",
            filename=f"epoch_to_max_acc_by_neg_conj_{self.name}.png",
            label_mode="complexity",
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
        if neg_df is None:
            out_dir_path = pathlib.Path(__file__).resolve().parent / "outputs"
            csv_path = out_dir_path / f"negation_analysis_{self.name}.csv"
            if not csv_path.is_file():
                print(f"[WARN] Negation analysis not found: {csv_path}")
                return
            neg_df = pd.read_csv(csv_path)

        filename = f"negation_scores_{mode}_{self.name}.png"
        self._plot_negation_scores_generic(
            neg_df,
            mode=mode,
            filename=filename,
            label_mode=label_mode,
            out_dir=out_dir,
        )

    def summarize_negation_by_complexity(self, neg_df=None, *, mode="greedy", threshold=0.9, out_dir="outputs"):
        if neg_df is None:
            out_dir_path = pathlib.Path(__file__).resolve().parent / "outputs"
            csv_path = out_dir_path / f"negation_analysis_{self.name}.csv"
            if not csv_path.is_file():
                print(f"[WARN] Negation analysis not found: {csv_path}")
                return None
            neg_df = pd.read_csv(csv_path)

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
            N = int(cfg.get("properties"))
            has_neg = not cfg.get("no_negation", False)
            M = 2 * N if has_neg else N
            A = 2 * N + 1
            theoretical_min = math.log(M, A) if M > 1 else 1.0
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

        fig = plt.figure(figsize=(10, 6))
        for key, curves in groups.items():
            curves = np.array(curves)
            mean_y = curves.mean(axis=0)
            median_y = np.median(curves, axis=0)
            min_y = curves.min(axis=0)
            max_y = curves.max(axis=0)

            label = ", ".join(f"{k}={v}" for k, v in zip(group_by, key))
            p = plt.plot(epochs, mean_y, label=f"{label} (mean)", alpha=0.9, linewidth=2)
            color = p[0].get_color()
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
