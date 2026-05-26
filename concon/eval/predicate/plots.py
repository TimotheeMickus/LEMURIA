import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import math
import re
from scipy.stats import pearsonr, spearmanr, ttest_ind
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
            voc_penalty = dp["config"].get("voc_penalty", dp["config"].get("voc_pen"))
            reaper_interval = dp["config"].get("beth_reaper_step", dp["config"].get("reaper_step"))
            # Find max accuracy/perf and the first epoch where those maxima are achieved.
            max_acc = eval_df["eval/accuracy"].max()
            best_epoch_acc = eval_df.loc[eval_df["eval/accuracy"] >= (max_acc - 0.001), "epoch"].min() + 1 # epochs are 0-index

            results.append({
                    'complexity': int(complexity),
                    'properties': properties,
                    'hidden_size': hidden,
                    'negation': "-neg" if dp["config"].get("no_negation", False) else "+neg",
                    'conjunction': "-conj" if dp["config"].get("no_conjunction", False) else "+conj",
                    'voc_penalty': voc_penalty,
                    'reaper_interval': reaper_interval,
                    'epochs_to_max_acc': best_epoch_acc,
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
            stats = (
                plot_df.groupby("complexity")["epochs_to_max_acc"]
                .agg(mean="mean", lo="min", hi="max")
                .reset_index()
            )
            plt.fill_between(stats["complexity"], stats["lo"], stats["hi"], alpha=0.15)
            plt.plot(stats["complexity"], stats["mean"], marker="o", label="max accuracy")
        else:
            long_df = plot_df.copy()
            long_df = long_df[long_df["epochs_to_max_acc"].notna()]
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
            for hval, grp in long_df.groupby(hue):
                stats = (
                    grp.groupby("complexity")["epochs_to_max_acc"]
                    .agg(mean="mean", lo="min", hi="max")
                    .reset_index()
                )
                label = f"{hval}"
                color = hue_color.get(hval)
                linestyle, marker = "-", "o"
                plt.fill_between(stats["complexity"], stats["lo"], stats["hi"], alpha=0.15, color=color if color is not None else None)
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
        plt.close(fig)

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
        plt.close(fig)

    def plot_epochs_to_max(self, plot_df, out_dir="plots"):
        self._plot_epochs_generic(
            plot_df,
            hue=None,
            filename=f"epoch_to_max_metrics_{self.name}.png",
            label_mode="full",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_voc_penalty(self, plot_df, out_dir="plots"):
        if "voc_penalty" not in plot_df.columns:
            print("[INFO] Convergence-by-voc_penalty plot skipped (missing voc_penalty).")
            return

        plot_df = plot_df.copy()
        plot_df["voc_penalty"] = pd.to_numeric(plot_df["voc_penalty"], errors="coerce")
        plot_df = plot_df[plot_df["voc_penalty"].notna()]
        if plot_df.empty:
            print("[INFO] Convergence-by-voc_penalty plot skipped (no valid voc_penalty values).")
            return

        fig = plt.figure(figsize=(10, 6))
        stats = (
            plot_df.groupby("voc_penalty")["epochs_to_max_acc"]
            .agg(mean="mean", lo="min", hi="max")
            .reset_index()
            .sort_values("voc_penalty")
        )
        plt.fill_between(stats["voc_penalty"], stats["lo"], stats["hi"], alpha=0.15)
        plt.plot(stats["voc_penalty"], stats["mean"], marker="o", label="max accuracy")

        xticks = sorted(plot_df["voc_penalty"].dropna().unique())
        pos_ticks = [x for x in xticks if x > 0]
        if len(pos_ticks) > 0:
            if any(x <= 0 for x in xticks):
                plt.xscale("symlog", linthresh=min(pos_ticks) / 2.0, base=10)
            else:
                plt.xscale("log", base=10)
        if xticks:
            if len(xticks) > 10:
                idx = np.unique(np.linspace(0, len(xticks) - 1, num=10).round().astype(int))
                shown_ticks = [xticks[i] for i in idx]
            else:
                shown_ticks = xticks
            labels = [("0" if abs(float(x)) < 1e-15 else f"{float(x):.12f}".rstrip("0").rstrip(".")) for x in shown_ticks]
            rot = 0 if len(xticks) <= 8 else 45
            plt.xticks(shown_ticks, labels, rotation=rot, ha="right" if rot else "center")

        plt.title("epochs until convergence by voc_penalty", fontweight="bold")
        plt.xlabel("voc_penalty")
        plt.ylabel("epochs to max")
        plt.grid(True, alpha=0.3)
        plt.legend()

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"epoch_to_max_metrics_by_voc_penalty_{self.name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

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

    def plot_epochs_to_max_by_reaper_interval(self, plot_df, out_dir="plots"):
        if "reaper_interval" not in plot_df.columns:
            print("[INFO] Convergence-by-reaper plot skipped (missing reaper_interval).")
            return
        sub = plot_df.copy()
        sub["reaper_interval"] = pd.to_numeric(sub["reaper_interval"], errors="coerce")
        sub = sub[sub["reaper_interval"].notna()]
        if sub.empty:
            print("[INFO] Convergence-by-reaper plot skipped (no valid reaper_interval values).")
            return
        self._plot_epochs_generic(
            sub,
            hue="reaper_interval",
            filename=f"epoch_to_max_metrics_by_reaper_interval_{self.name}.png",
            label_mode="full",
            out_dir=out_dir,
        )

    def plot_epochs_to_max_by_reaper_interval_per_complexity(self, plot_df, out_dir="plots"):
        required = {"reaper_interval", "complexity", "epochs_to_max_acc"}
        if not required.issubset(plot_df.columns):
            print("[INFO] Convergence-by-reaper-per-complexity skipped (missing columns).")
            return

        sub = plot_df.copy()
        sub["reaper_interval"] = pd.to_numeric(sub["reaper_interval"], errors="coerce")
        sub["complexity"] = pd.to_numeric(sub["complexity"], errors="coerce")
        sub = sub[sub["reaper_interval"].notna() & sub["complexity"].notna()]
        if sub.empty:
            print("[INFO] Convergence-by-reaper-per-complexity skipped (no valid values).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        complexities = sorted(sub["complexity"].unique())
        for c in complexities:
            g = sub[sub["complexity"] == c]
            if g.empty:
                continue

            fig = plt.figure(figsize=(10, 6))
            gg = g[g["epochs_to_max_acc"].notna()]
            if gg.empty:
                plt.close(fig)
                continue
            stats = (
                gg.groupby("reaper_interval")["epochs_to_max_acc"]
                .agg(mean="mean", lo="min", hi="max")
                .reset_index()
                .sort_values("reaper_interval")
            )
            plt.fill_between(stats["reaper_interval"], stats["lo"], stats["hi"], alpha=0.15)
            plt.plot(stats["reaper_interval"], stats["mean"], marker="o", label="max accuracy")

            ticks = sorted(g["reaper_interval"].dropna().unique())
            if ticks:
                plt.xscale("log")
                labels = [str(int(x)) if float(x).is_integer() else f"{x:g}" for x in ticks]
                plt.xticks(ticks, labels)

            props = (
                g["properties"].dropna().astype(str).unique()[0]
                if ("properties" in g.columns and g["properties"].dropna().size > 0)
                else str(int(c))
            )
            plt.title(f"epochs until convergence by reaper interval | properties={props}", fontweight="bold")
            plt.xlabel("reaper interval")
            plt.ylabel("epochs to max")
            plt.grid(True, alpha=0.3)
            plt.legend()

            fig.tight_layout()
            fig.savefig(
                out_dir / f"epoch_to_max_metrics_by_reaper_interval_props_{props}_{self.name}.png",
                dpi=150,
            )
            plt.close(fig)

    def plot_time_to_max_accuracy(self, out_dir="plots"):
        plot_df = self._build_epochs_df()
        if plot_df is None:
            return
        self.plot_epochs_to_max(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_hidden(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_negation(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_conjunction(plot_df, out_dir=out_dir)
        self.plot_epochs_to_max_by_negation_and_conjunction(plot_df, out_dir=out_dir)
        summary = plot_df.groupby('complexity')[['epochs_to_max_acc']].agg(['mean', 'std', 'count'])
        print(summary)

    def _build_eval_accuracy_df(self):
        rows = []
        for dp in self.datapoints:
            eval_df = dp.get("evaluation")
            if eval_df is None or eval_df.empty:
                continue
            if "epoch" not in eval_df.columns or "eval/accuracy" not in eval_df.columns:
                continue
            cfg = dp.get("config", {})
            reaper_interval = cfg.get("beth_reaper_step")
            if reaper_interval is None:
                reaper_interval = cfg.get("reaper_step")
            voc_penalty = cfg.get("voc_penalty", cfg.get("voc_pen"))
            run_name = str(dp.get("run_name", ""))
            for _, row in eval_df.iterrows():
                rows.append({
                    "run_name": run_name,
                    "epoch": row.get("epoch"),
                    "eval_accuracy": row.get("eval/accuracy"),
                    "complexity": cfg.get("num_predicates"),
                    "properties": str(cfg.get("properties")),
                    "reaper_interval": reaper_interval,
                    "voc_penalty": voc_penalty,
                })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df["eval_accuracy"] = pd.to_numeric(df["eval_accuracy"], errors="coerce")
        df = df.dropna(subset=["epoch", "eval_accuracy"])
        if df.empty:
            return None
        return df

    def plot_eval_accuracy_over_epochs(self, *, group_col=None, out_dir="plots"):
        df = self._build_eval_accuracy_df()
        if df is None or df.empty:
            print("[INFO] eval/accuracy curve skipped (no evaluation data).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(10, 6))
        suffix = "overall"
        title = "eval/accuracy over epochs"

        if group_col is None:
            stats = (
                df.groupby("epoch")["eval_accuracy"]
                .agg(median="median", q25=lambda x: np.nanquantile(x, 0.25), q75=lambda x: np.nanquantile(x, 0.75))
                .reset_index()
                .sort_values("epoch")
            )
            plt.fill_between(stats["epoch"], stats["q25"], stats["q75"], alpha=0.18, color="#1f77b4")
            plt.plot(stats["epoch"], stats["median"], color="#1f77b4", linewidth=2.2, label="median")
        else:
            if group_col not in df.columns:
                print(f"[INFO] eval/accuracy curve skipped (missing grouping column: {group_col}).")
                plt.close(fig)
                return
            sub = df[df[group_col].notna()].copy()
            if sub.empty:
                print(f"[INFO] eval/accuracy curve skipped (no values for {group_col}).")
                plt.close(fig)
                return

            gvals = sorted(pd.to_numeric(sub[group_col], errors="coerce").dropna().unique())
            if len(gvals) == 0:
                print(f"[INFO] eval/accuracy curve skipped (non-numeric {group_col}).")
                plt.close(fig)
                return

            palette = sns.color_palette("viridis", n_colors=max(1, len(gvals)))
            for i, g in enumerate(gvals):
                grp = sub[pd.to_numeric(sub[group_col], errors="coerce") == g]
                stats = (
                    grp.groupby("epoch")["eval_accuracy"]
                    .agg(median="median", q25=lambda x: np.nanquantile(x, 0.25), q75=lambda x: np.nanquantile(x, 0.75))
                    .reset_index()
                    .sort_values("epoch")
                )
                c = palette[i]
                plt.fill_between(stats["epoch"], stats["q25"], stats["q75"], alpha=0.12, color=c)
                label = f"{g:g}" if group_col == "voc_penalty" else str(int(g) if float(g).is_integer() else g)
                plt.plot(stats["epoch"], stats["median"], color=c, linewidth=2.0, label=label)

            suffix = f"by_{group_col}"
            title = f"eval/accuracy over epochs by {group_col}"
            plt.legend(title=group_col, ncol=2)

        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel("eval/accuracy")
        plt.grid(True, alpha=0.3)
        plt.ylim(0.0, 1.0)
        fig.tight_layout()
        fig.savefig(out_dir / f"eval_accuracy_over_epochs_{suffix}_{self.name}.png", dpi=150)
        plt.close(fig)

    def plot_eval_accuracy_over_epochs_by_reaper_and_complexity(self, *, out_dir="plots"):
        df = self._build_eval_accuracy_df()
        if df is None or df.empty:
            print("[INFO] eval/accuracy by reaper+complexity skipped (no evaluation data).")
            return
        if "reaper_interval" not in df.columns or "complexity" not in df.columns:
            print("[INFO] eval/accuracy by reaper+complexity skipped (missing columns).")
            return

        df = df.copy()
        df["reaper_interval"] = pd.to_numeric(df["reaper_interval"], errors="coerce")
        df["complexity"] = pd.to_numeric(df["complexity"], errors="coerce")
        df = df.dropna(subset=["reaper_interval", "complexity"])
        df = df[df["reaper_interval"] > 0]
        if df.empty:
            print("[INFO] eval/accuracy by reaper+complexity skipped (no valid values).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        complexities = sorted(df["complexity"].unique())
        for c in complexities:
            sub = df[df["complexity"] == c]
            if sub.empty:
                continue

            reapers = sorted(sub["reaper_interval"].unique())
            palette = sns.color_palette("viridis", n_colors=max(1, len(reapers)))

            fig = plt.figure(figsize=(10, 6))
            for i, r in enumerate(reapers):
                grp = sub[sub["reaper_interval"] == r]
                stats = (
                    grp.groupby("epoch")["eval_accuracy"]
                    .agg(median="median", q25=lambda x: np.nanquantile(x, 0.25), q75=lambda x: np.nanquantile(x, 0.75))
                    .reset_index()
                    .sort_values("epoch")
                )
                ccol = palette[i]
                plt.fill_between(stats["epoch"], stats["q25"], stats["q75"], alpha=0.10, color=ccol)
                rlab = str(int(r)) if float(r).is_integer() else f"{r:g}"
                plt.plot(stats["epoch"], stats["median"], color=ccol, linewidth=2.0, label=rlab)

            props = (
                sub["properties"].dropna().astype(str).unique()[0]
                if ("properties" in sub.columns and sub["properties"].dropna().size > 0)
                else str(int(c))
            )
            plt.title(f"eval/accuracy over epochs by reaper interval | properties={props}")
            plt.xlabel("epoch")
            plt.ylabel("eval/accuracy")
            plt.grid(True, alpha=0.3)
            plt.ylim(0.0, 1.0)
            plt.legend(title="reaper_interval", ncol=2)

            fig.tight_layout()
            fig.savefig(out_dir / f"eval_accuracy_over_epochs_by_reaper_interval_props_{props}_{self.name}.png", dpi=150)
            plt.close(fig)

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
        plt.close(fig)

    def _extract_run_id(self, run_name):
        m = re.search(r"__run=(\d+)", str(run_name))
        return int(m.group(1)) if m else np.nan

    def _build_run_meta_df(self):
        rows = []
        for dp in self.datapoints:
            cfg = dp.get("config", {})
            reaper_interval = cfg.get("beth_reaper_step")
            if reaper_interval is None:
                reaper_interval = cfg.get("reaper_step")
            rows.append({
                "run_name": str(dp.get("run_name", "")),
                "complexity": cfg.get("num_predicates"),
                "hidden_size": cfg.get("hidden_size"),
                "properties": str(cfg.get("properties")),
                "no_negation": cfg.get("no_negation", False),
                "no_conjunction": cfg.get("no_conjunction", False),
                "reaper_interval": reaper_interval,
                "voc_penalty": cfg.get("voc_penalty", cfg.get("voc_pen")),
            })
        return pd.DataFrame(rows).drop_duplicates(subset=["run_name"])

    def _build_topsim_df(self):
        rows = []
        topsim_col_candidates = [
            "eval/topsim_intensional_norm_levenshtein",
            "topsim_intensional_norm_levenshtein",
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

    def _condition_label(self, cfg):
        no_conj = bool(cfg.get("no_conjunction", False))
        no_neg = bool(cfg.get("no_negation", False))
        if not no_conj:
            return "n-n"
        return "n-neg" if no_neg else "n+neg"

    def _build_complexity_memory_run_df(self):
        rows = []
        topsim_col_candidates = [
            "eval/topsim_intensional_norm_levenshtein",
            "topsim_intensional_norm_levenshtein",
            "eval/topsim_intensional_levenshtein",
            "topsim_intensional_levenshtein",
        ]
        for dp in self.datapoints:
            cfg = dp.get("config", {})
            eval_df = dp.get("evaluation")
            if eval_df is None or eval_df.empty:
                continue
            if "epoch" not in eval_df.columns:
                continue

            eval_work = eval_df.copy()
            eval_work["epoch"] = pd.to_numeric(eval_work["epoch"], errors="coerce")
            eval_work = eval_work[eval_work["epoch"].notna()]
            if eval_work.empty:
                continue

            max_acc = np.nan
            epochs_to_max = np.nan
            best_idx = eval_work.index[-1]
            if "eval/accuracy" in eval_work.columns:
                acc = pd.to_numeric(eval_work["eval/accuracy"], errors="coerce")
                valid = acc.notna()
                if valid.any():
                    max_acc = float(acc[valid].max())
                    hit = acc >= (max_acc - 1e-3)
                    if hit.any():
                        hit_epochs = pd.to_numeric(eval_work.loc[hit, "epoch"], errors="coerce")
                        if hit_epochs.notna().any():
                            epochs_to_max = float(hit_epochs.min()) + 1.0
                            first_hit_idx = eval_work.loc[hit].sort_values("epoch").index[0]
                            best_idx = first_hit_idx

            best_row = eval_work.loc[best_idx]
            topsim = np.nan
            for c in topsim_col_candidates:
                if c in eval_work.columns:
                    topsim = pd.to_numeric(best_row.get(c), errors="coerce")
                    break

            msg_eff = np.nan
            if "eval/msg_length" in eval_work.columns:
                msg_len = pd.to_numeric(best_row.get("eval/msg_length"), errors="coerce")
                N = cfg.get("num_predicates")
                A = cfg.get("base_alphabet_size")
                if pd.notna(msg_len) and N is not None and A is not None:
                    Nf = float(N)
                    Af = float(A)
                    # num_predicates already stores the real predicate count M for all
                    # conditions: n for n-neg, 2n for n+neg, n²+2n for n-n.
                    Mf = Nf
                    if Mf > 1.0 and Af > 1.0:
                        theo_min = math.log(Mf, Af)
                        if theo_min > 0.0:
                            msg_eff = float(msg_len) / theo_min

            rows.append({
                "run_name": str(dp.get("run_name", "")),
                "condition": self._condition_label(cfg),
                "complexity": cfg.get("num_predicates"),
                "hidden_size": cfg.get("hidden_size"),
                "properties": str(cfg.get("properties")),
                "topsim": float(topsim) if pd.notna(topsim) else np.nan,
                "max_acc": max_acc,
                "epochs_to_max_acc": epochs_to_max,
                "msg_eff": msg_eff,
            })

        if not rows:
            return None
        df = pd.DataFrame(rows)
        for c in ["complexity", "hidden_size", "topsim", "max_acc", "epochs_to_max_acc", "msg_eff"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _ttest_p(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return np.nan
        _, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(p) if np.isfinite(p) else np.nan

    def _build_msg_eff_over_epochs_df(self):
        """Per-epoch inverse compression ratio for n+neg and n-neg conditions."""
        rows = []
        for dp in self.datapoints:
            cfg = dp.get("config", {})
            eval_df = dp.get("evaluation")
            if eval_df is None or eval_df.empty:
                continue
            if "eval/msg_length" not in eval_df.columns or "epoch" not in eval_df.columns:
                continue
            cond = self._condition_label(cfg)
            if cond not in ("n+neg", "n-neg"):
                continue
            N = cfg.get("num_predicates")
            A = cfg.get("base_alphabet_size")
            hidden = cfg.get("hidden_size")
            if N is None or A is None:
                continue
            Nf, Af = float(N), float(A)
            if Nf <= 1.0 or Af <= 1.0:
                continue
            # num_predicates already stores the real count M (2n for n+neg, n for n-neg)
            Mf = Nf
            theo_min = math.log(Mf, Af)
            if theo_min <= 0.0:
                continue
            for _, row in eval_df.iterrows():
                ep = row.get("epoch")
                ml = row.get("eval/msg_length")
                if pd.isna(ep) or pd.isna(ml):
                    continue
                rows.append({
                    "condition": cond,
                    "hidden_size": hidden,
                    "epoch": int(float(ep)),
                    "inv_compression": float(ml) / theo_min,
                    "run_name": str(dp.get("run_name", "")),
                })
        if not rows:
            return None
        out = pd.DataFrame(rows)
        out["epoch"] = pd.to_numeric(out["epoch"], errors="coerce")
        out["inv_compression"] = pd.to_numeric(out["inv_compression"], errors="coerce")
        return out.dropna(subset=["epoch", "inv_compression"])

    def _write_latex_table(self, path, df, *, caption="", label="tab:table", col_fmt=None,
                           col_headers=None):
        """Write a LaTeX booktabs table (.tex) from a DataFrame."""
        cols = list(df.columns)
        headers = col_headers if col_headers is not None else cols
        if col_fmt is None:
            col_fmt = "l" + "r" * (len(cols) - 1)

        def _cell(v):
            if isinstance(v, float) and np.isnan(v):
                return r"\textemdash"
            if isinstance(v, float):
                return f"{v:.3g}"
            return str(v).replace("_", r"\_").replace("&", r"\&")

        header = " & ".join(_cell(c) for c in headers) + r" \\"
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            f"\\begin{{tabular}}{{{col_fmt}}}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for _, row in df.iterrows():
            lines.append(" & ".join(_cell(v) for v in row) + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
            "",
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines))

    def _write_pgfplots_tex(self, path, *, series, axis_opts=None, caption="", label="fig:plot"):
        """
        Write a standalone PGFPlots figure (.tex).
        series: list of dicts:
          type  = 'coords' | 'hline'
          coords = [(x,y), ...]    (for 'coords')
          y      = float           (for 'hline')
          opts   = pgfplots option string, e.g. '+[mark=none,thick]'
          legend = str | None      (None → no \\addlegendentry)
        """
        def _fmt(v):
            v = float(v)
            return str(int(v)) if v == int(v) else f"{v:.6g}"

        ao = list(axis_opts or [])
        lines = [
            r"\begin{figure}[t]",
            r"\centering",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"    width=\linewidth,",
            r"    height=0.55\linewidth,",
            r"    grid=both,",
            r"    legend style={at={(0.5,1.02)},anchor=south,legend columns=3},",
        ]
        for opt in ao:
            lines.append(f"    {opt},")
        lines += [r"]", ""]
        for s in series:
            opts = s.get("opts", "+[mark=none,thick]")
            legend = s.get("legend")
            if s.get("type") == "hline":
                lines.append(f"\\addplot{opts} {{{_fmt(s['y'])}}};")
            elif s.get("type") == "fill_between":
                # Closed polygon: upper coords L→R, lower coords R→L
                upper = [(x, y) for x, y in s.get("upper", [])
                         if np.isfinite(float(x)) and np.isfinite(float(y))]
                lower = [(x, y) for x, y in s.get("lower", [])
                         if np.isfinite(float(x)) and np.isfinite(float(y))]
                poly = upper + list(reversed(lower))
                lines.append(f"\\addplot{opts} coordinates {{")
                for x, y in poly:
                    lines.append(f"    ({_fmt(x)},{_fmt(y)})")
                lines.append("} -- cycle;")
            else:
                coords = [
                    (x, y) for x, y in s.get("coords", [])
                    if np.isfinite(float(x)) and np.isfinite(float(y))
                ]
                lines.append(f"\\addplot{opts} coordinates {{")
                for x, y in coords:
                    lines.append(f"    ({_fmt(x)},{_fmt(y)})")
                lines.append("};")
            if legend is not None:
                lines.append(f"\\addlegendentry{{{legend}}}")
            lines.append("")
        lines += [
            r"\end{axis}",
            r"\end{tikzpicture}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{figure}",
            "",
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines))

    def plot_may_complexity_suite(self, *, out_dir="plots_may"):
        _steps = [
            "build run df",
            "topsim vs complexity (3×3 grid)",
            "inverse compression ratio over epochs",
            "training efficiency (hidden=48 plot)",
            "training efficiency (summary table by hidden size)",
            "negation helps learning (hidden=48, delta)",
            "message efficiency by negation",
        ]
        _n = len(_steps)
        def _progress(i, label):
            print(f"  [plots_may {i}/{_n}] {label} ...", flush=True)

        _progress(1, _steps[0])
        df = self._build_complexity_memory_run_df()
        if df is None or df.empty:
            print("[INFO] plots_may skipped (no usable run-level data).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cond_order = ["n+neg", "n-neg", "n-n"]
        cond_labels = {"n+neg": "unary + negation", "n-neg": "unary, no negation", "n-n": "conjunction"}
        cond_colors = {"n+neg": "#1f77b4", "n-neg": "#e07a2d", "n-n": "#2ca02c"}
        hidden_vals = sorted(pd.to_numeric(df["hidden_size"], errors="coerce").dropna().unique())
        h_pal = sns.color_palette("viridis", n_colors=max(1, len(hidden_vals)))
        h_color = {h: h_pal[i] for i, h in enumerate(hidden_vals)}

        df.sort_values(["condition", "hidden_size", "complexity", "run_name"]).to_csv(
            out_dir / f"may_raw_run_metrics_{self.name}.csv", index=False
        )

        def _set_log2_xticks(ax, vals):
            arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().unique()
            ticks = sorted(arr)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [str(int(t)) if float(t).is_integer() else f"{t:g}" for t in ticks],
                rotation=45, ha="right", fontsize=8,
            )

        def _add_log_trend(ax, raw_x, raw_y, color):
            xlog = np.log2(np.asarray(raw_x, dtype=float) + 1e-9)
            yv = np.asarray(raw_y, dtype=float)
            mask = np.isfinite(xlog) & np.isfinite(yv)
            if mask.sum() < 3 or np.ptp(xlog[mask]) < 1e-9:
                return
            a, b = np.polyfit(xlog[mask], yv[mask], 1)
            xg = np.logspace(np.log2(np.asarray(raw_x)[mask].min()),
                             np.log2(np.asarray(raw_x)[mask].max()), 120, base=2.0)
            ax.plot(xg, a * np.log2(xg) + b, "--", color=color, linewidth=1.3, alpha=0.60)

        def _ann(r, p, n):
            r_txt = f"ρ={r:.2f}" if np.isfinite(r) else "ρ=n/a"
            return f"{r_txt}, {self._format_p_value(p)}, n={n}"

        # ----------------------------------------------------------------
        # 1. TOPSIM VS COMPLEXITY — 3 rows (one per hidden size)
        # ----------------------------------------------------------------
        _progress(2, _steps[1])
        tdf = df.dropna(subset=["topsim", "complexity"]).copy()

        corr_rows = []
        for cond in cond_order:
            sub = tdf[tdf["condition"] == cond]
            x = sub["complexity"].to_numpy(dtype=float)
            y = sub["topsim"].to_numpy(dtype=float)
            r, p = self._spearman_r_p(x, y)
            corr_rows.append({"level": "condition", "condition": cond, "hidden_size": np.nan,
                               "n": int(len(sub)), "rho": r, "p": p})
            for h, g in sub.groupby("hidden_size"):
                xh = g["complexity"].to_numpy(dtype=float)
                yh = g["topsim"].to_numpy(dtype=float)
                rh, ph = self._spearman_r_p(xh, yh)
                corr_rows.append({"level": "condition_hidden", "condition": cond, "hidden_size": h,
                                   "n": int(len(g)), "rho": rh, "p": ph})
        pd.DataFrame(corr_rows).to_csv(
            out_dir / f"may_topsim_complexity_correlations_{self.name}.csv", index=False)
        tdf.sort_values(["condition", "hidden_size", "complexity", "run_name"]).to_csv(
            out_dir / f"may_topsim_complexity_points_{self.name}.csv", index=False)

        agg_top_h = (
            tdf.groupby(["condition", "hidden_size", "complexity"])["topsim"]
            .agg(topsim_median="median", topsim_std="std", topsim_n="count")
            .reset_index()
        )
        agg_top_h.to_csv(out_dir / f"may_topsim_by_condition_hidden_complexity_{self.name}.csv", index=False)

        def _nn_to_n_sq(c):
            # num_predicates = n² + 2n  →  n = -1 + sqrt(1 + c)  →  n² = n*n
            n = round(-1.0 + math.sqrt(1.0 + float(c)))
            return n * n

        def _plot_x(cond, complexity_series):
            """Return x-positions for plotting: n² for n-n, raw complexity otherwise."""
            if cond == "n-n":
                return complexity_series.apply(_nn_to_n_sq)
            return complexity_series

        # 3 rows (hidden size) × 3 columns (condition) — scatter + trend + r/p annotation
        ncols = len(cond_order)
        nrows = max(1, len(hidden_vals))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 4.0 * nrows),
            sharex="col", sharey=True,
            squeeze=False,
        )
        for j, cond in enumerate(cond_order):
            cond_sub = tdf[tdf["condition"] == cond].copy()
            cond_sub["x_plot"] = _plot_x(cond, cond_sub["complexity"])
            cond_x_vals = sorted(cond_sub["x_plot"].dropna().unique())
            color = cond_colors[cond]
            for i, h in enumerate(hidden_vals):
                ax = axes[i][j]
                cell = cond_sub[cond_sub["hidden_size"] == h]
                x_raw = cell["x_plot"].to_numpy(dtype=float)
                xlog = np.log2(x_raw + 1e-9)
                y = cell["topsim"].to_numpy(dtype=float)
                r, p = self._spearman_r_p(x_raw, y)
                if not cell.empty:
                    ax.scatter(cell["x_plot"], cell["topsim"],
                               color=color, alpha=0.55, s=22, zorder=3)
                    mask = np.isfinite(xlog) & np.isfinite(y)
                    if mask.sum() >= 3 and np.ptp(xlog[mask]) > 1e-9:
                        a, b = np.polyfit(xlog[mask], y[mask], 1)
                        xg = np.logspace(np.log2(cell["x_plot"].min()),
                                         np.log2(cell["x_plot"].max()), 120, base=2.0)
                        ax.plot(xg, a * np.log2(xg) + b, "-", color=color,
                                linewidth=1.8, alpha=0.8)
                r_txt = f"ρ={r:.2f}" if np.isfinite(r) else "ρ=n/a"
                cell_title = f"{r_txt}, {self._format_p_value(p)}"
                if i == 0:
                    ax.set_title(
                        f"{cond_labels.get(cond, cond)}\n{cell_title}",
                        fontsize=10, fontweight="bold",
                    )
                else:
                    ax.set_title(cell_title, fontsize=9)
                ax.grid(True, alpha=0.25)
                ax.set_ylim(bottom=0)
                if j == 0:
                    ax.set_ylabel(f"hidden = {int(h)}\ntopographic similarity", fontsize=9)
                if i == nrows - 1:
                    ax.set_xlabel("predicate space size  (n² for conjunction)", fontsize=9)
            # set x-ticks per column using the (possibly remapped) x values
            if cond_x_vals:
                axes[-1][j].set_xscale("log", base=2)
                axes[-1][j].set_xticks(cond_x_vals)
                axes[-1][j].set_xticklabels(
                    [str(int(t)) if float(t).is_integer() else f"{t:g}" for t in cond_x_vals],
                    rotation=45, ha="right", fontsize=7,
                )
        fig.suptitle("Topographic similarity vs. predicate space size",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"may_topsim_vs_complexity_{self.name}.png", dpi=150)
        plt.close(fig)

        # Topsim LaTeX groupplot (pgfplots)
        _pgf_cond_rgb = {
            "n+neg": "31,119,180",   # #1f77b4
            "n-neg": "224,122,45",   # #e07a2d
            "n-n":   "44,160,44",    # #2ca02c
        }
        _pgf_cond_name = {"n+neg": "colnpneg", "n-neg": "colnneg", "n-n": "colnn"}
        tex_lines = [
            r"\begin{figure}[t]",
            r"\centering",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tikzpicture}",
        ]
        for cond, rgb in _pgf_cond_rgb.items():
            tex_lines.append(f"\\definecolor{{{_pgf_cond_name[cond]}}}{{RGB}}{{{rgb}}}")
        tex_lines += [
            r"\pgfplotsset{every axis/.append style={",
            r"    grid=both, ymin=0,",
            r"    x tick label style={rotate=45,anchor=east,font=\scriptsize},",
            r"    title style={font=\small},",
            r"    label style={font=\small},",
            r"    tick label style={font=\scriptsize},",
            r"}}",
            r"\begin{groupplot}[",
            r"    group style={",
            f"        group size={len(cond_order)} by {len(hidden_vals)},",
            r"        xlabels at=edge bottom,",
            r"        ylabels at=edge left,",
            r"        horizontal sep=0.6cm,",
            r"        vertical sep=0.9cm,",
            r"    },",
            r"    width=0.38\textwidth,",
            r"    height=0.30\textwidth,",
            r"]",
            "",
        ]
        for i, h in enumerate(hidden_vals):
            for j, cond in enumerate(cond_order):
                cond_sub_pgf = tdf[tdf["condition"] == cond].copy()
                cond_sub_pgf["x_plot"] = _plot_x(cond, cond_sub_pgf["complexity"])
                cell_pgf = cond_sub_pgf[cond_sub_pgf["hidden_size"] == h]
                cond_x_vals_pgf = sorted(cond_sub_pgf["x_plot"].dropna().unique())
                x_raw_pgf = cell_pgf["x_plot"].to_numpy(dtype=float)
                xlog_pgf = np.log2(x_raw_pgf + 1e-9)
                yv_pgf = cell_pgf["topsim"].to_numpy(dtype=float)
                r_pgf, p_pgf = self._spearman_r_p(x_raw_pgf, yv_pgf)
                r_str = f"$\\rho$={r_pgf:.2f}" if np.isfinite(r_pgf) else "$\\rho$=n/a"
                p_str = self._format_p_value(p_pgf)
                title_inner = f"{r_str}, {p_str}"
                if i == 0:
                    cond_lbl_pgf = cond_labels.get(cond, cond).replace("&", r"\&")
                    title_str = f"{cond_lbl_pgf}\\\\{title_inner}"
                else:
                    title_str = title_inner
                xtick_str = ",".join(
                    str(int(v)) if float(v).is_integer() else f"{v:g}"
                    for v in cond_x_vals_pgf)
                next_opts = [
                    f"title={{{title_str}}}",
                    "xmode=log, log basis x=2",
                    f"xtick={{{xtick_str}}}",
                    f"xticklabels={{{xtick_str}}}",
                ]
                if j == 0:
                    next_opts.append(f"ylabel={{hidden={int(h)}, topsim}}")
                if i == len(hidden_vals) - 1:
                    next_opts.append(r"xlabel={predicate space size}")
                tex_lines.append(r"\nextgroupplot[" + ", ".join(next_opts) + "]")
                if not cell_pgf.empty:
                    sc_pts = [
                        (float(x), float(y))
                        for x, y in zip(cell_pgf["x_plot"], cell_pgf["topsim"])
                        if np.isfinite(float(x)) and np.isfinite(float(y))
                    ]
                    if sc_pts:
                        col_name = _pgf_cond_name[cond]
                        tex_lines.append(
                            f"\\addplot[only marks, mark=*, mark size=1.2pt,"
                            f" color={col_name}, opacity=0.55, forget plot] coordinates {{")
                        for xp, yp in sc_pts:
                            tex_lines.append(
                                f"    ({int(xp) if xp == int(xp) else f'{xp:g}'},{yp:.4f})")
                        tex_lines.append("};")
                    mask_pgf = np.isfinite(xlog_pgf) & np.isfinite(yv_pgf)
                    if mask_pgf.sum() >= 3 and np.ptp(xlog_pgf[mask_pgf]) > 1e-9:
                        a_c, b_c = np.polyfit(xlog_pgf[mask_pgf], yv_pgf[mask_pgf], 1)
                        xg_pgf = np.logspace(
                            np.log2(float(cell_pgf["x_plot"].min())),
                            np.log2(float(cell_pgf["x_plot"].max())),
                            30, base=2.0)
                        tex_lines.append(
                            f"\\addplot[thick, color={_pgf_cond_name[cond]},"
                            f" forget plot] coordinates {{")
                        for xp in xg_pgf:
                            yp = a_c * np.log2(xp) + b_c
                            tex_lines.append(
                                f"    ({xp:.4g},{max(0.0, yp):.4f})")
                        tex_lines.append("};")
                tex_lines.append("")
        tex_lines += [
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            r"}",  # end \resizebox
            (r"\caption{Topographic similarity vs.\ predicate space size. "
             r"Columns: conditions (unary+neg, unary, conjunction). "
             r"Rows: hidden sizes. "
             r"Scatter: individual runs; solid line: log-linear trend. "
             r"Annotation: Pearson $r$ and $p$-value. "
             r"For conjunction, the x-axis shows $n^2$ where $n$ is the number of base properties.}"),
            f"\\label{{fig:topsim_vs_complexity}}",
            r"\end{figure}",
            "",
        ]
        with open(out_dir / f"may_topsim_vs_complexity_{self.name}.tex", "w") as fh:
            fh.write("\n".join(tex_lines))

        # ----------------------------------------------------------------
        # 2. INVERSE COMPRESSION RATIO OVER EPOCHS — aggregated plot + per-hidden table
        # ----------------------------------------------------------------
        _progress(3, _steps[2])
        msg_ep_df = self._build_msg_eff_over_epochs_df()

        if msg_ep_df is not None and not msg_ep_df.empty:
            msg_ep_df.sort_values(["condition", "hidden_size", "epoch"]).to_csv(
                out_dir / f"may_inv_compression_over_epochs_{self.name}.csv", index=False)
            hidden_in_msg = sorted(
                pd.to_numeric(msg_ep_df["hidden_size"], errors="coerce").dropna().unique())

            # 3 panels: one per hidden size, n+neg vs n-neg
            n_ic = len(hidden_in_msg)
            fig_ic, axes_ic = plt.subplots(
                1, n_ic, figsize=(5.5 * n_ic, 4.5), sharey=True, squeeze=False)
            for i_ic, h_ic in enumerate(hidden_in_msg):
                ax_ic = axes_ic[0][i_ic]
                for cond_ic, color_ic in [("n+neg", "#1f77b4"), ("n-neg", "#e07a2d")]:
                    g_ic = (
                        msg_ep_df[(msg_ep_df["hidden_size"] == h_ic) &
                                  (msg_ep_df["condition"] == cond_ic)]
                        .groupby("epoch")["inv_compression"]
                        .agg(median="median",
                             q25=lambda x: np.nanquantile(x, 0.25),
                             q75=lambda x: np.nanquantile(x, 0.75))
                        .reset_index().sort_values("epoch")
                    )
                    if g_ic.empty:
                        continue
                    ep_ic = g_ic["epoch"].to_numpy()
                    med_ic = g_ic["median"].to_numpy()
                    ax_ic.fill_between(ep_ic, g_ic["q25"], g_ic["q75"],
                                       alpha=0.13, color=color_ic)
                    ax_ic.plot(ep_ic, med_ic, color=color_ic, linewidth=2.2,
                               label=cond_labels.get(cond_ic, cond_ic))
                    if len(ep_ic) >= 3:
                        z_ic = np.polyfit(ep_ic, med_ic, 1)
                        ax_ic.plot(ep_ic, np.poly1d(z_ic)(ep_ic), "--",
                                   color=color_ic, linewidth=1.3, alpha=0.60)
                ax_ic.axhline(1.0, color="#555555", linestyle="--",
                              linewidth=1.3, label="optimal")
                ax_ic.set_title(f"hidden = {int(h_ic)}", fontsize=10)
                ax_ic.set_xlabel("epoch", fontsize=9)
                if i_ic == 0:
                    ax_ic.set_ylabel("inverse compression ratio", fontsize=9)
                ax_ic.legend(fontsize=8, loc="upper left")
                ax_ic.grid(True, alpha=0.25)
            fig_ic.suptitle("Inverse compression ratio over training",
                            fontsize=13, fontweight="bold")
            fig_ic.tight_layout()
            fig_ic.savefig(
                out_dir / f"may_inv_compression_over_epochs_{self.name}.png", dpi=150)
            plt.close(fig_ic)

            # Summary table: one row per hidden size, gap trend = (+neg) − (−neg)
            tbl_rows = []
            for h_t in hidden_in_msg:
                trends_t = {}
                for cond_t in ["n+neg", "n-neg"]:
                    gh_t = (
                        msg_ep_df[(msg_ep_df["hidden_size"] == h_t) &
                                  (msg_ep_df["condition"] == cond_t)]
                        .groupby("epoch")["inv_compression"].median()
                        .reset_index().sort_values("epoch")
                    )
                    if gh_t.empty or len(gh_t) < 3:
                        trends_t[cond_t] = (np.nan, np.nan)
                        continue
                    ep_t = gh_t["epoch"].to_numpy()
                    med_t = gh_t["inv_compression"].to_numpy()
                    slope_t, intercept_t = np.polyfit(ep_t, med_t, 1)
                    trends_t[cond_t] = (float(slope_t), float(intercept_t))
                sp, bp = trends_t.get("n+neg", (np.nan, np.nan))
                sm, bm = trends_t.get("n-neg", (np.nan, np.nan))
                tbl_rows.append({
                    "hidden": int(h_t),
                    "delta_intercept": (round(bp - bm, 3)
                                        if np.isfinite(bp) and np.isfinite(bm) else np.nan),
                    "delta_slope": (round(sp - sm, 5)
                                    if np.isfinite(sp) and np.isfinite(sm) else np.nan),
                })
            tbl_df = pd.DataFrame(tbl_rows)
            tbl_df.to_csv(
                out_dir / f"may_inv_compression_summary_table_{self.name}.csv", index=False)
            self._write_latex_table(
                out_dir / f"may_inv_compression_summary_table_{self.name}.tex",
                tbl_df,
                col_headers=[
                    "hidden",
                    r"$\Delta$ intercept",
                    r"$\Delta$ slope",
                ],
                caption=(
                    r"Gap between the linear trends of the inverse compression ratio "
                    r"for games with and without negation: "
                    r"$(+\text{neg}) - (-\text{neg})$. "
                    r"$\Delta$ intercept = gap at epoch 0; "
                    r"$\Delta$ slope = change in gap per epoch "
                    r"(negative = gap narrowing, i.e.\ negation advantage grows)."
                ),
                label="tab:inv_compression_summary",
                col_fmt="crr",
            )
        else:
            print("[INFO] Inverse compression skipped (no eval/msg\_length epoch data).")

        # ----------------------------------------------------------------
        # 3. TRAINING EFFICIENCY — hidden=48, real predicate count x-axis
        # ----------------------------------------------------------------
        _progress(4, _steps[3])
        edf = df.dropna(subset=["epochs_to_max_acc", "complexity"]).copy()

        eff_corr = []
        for cond in cond_order:
            sub = edf[edf["condition"] == cond]
            x = np.log2(sub["complexity"].to_numpy(dtype=float) + 1e-9)
            y = sub["epochs_to_max_acc"].to_numpy(dtype=float)
            r, p = self._pearson_r_p(x, y)
            eff_corr.append({"condition": cond, "n": int(len(sub)), "r": r, "p": p})
        pd.DataFrame(eff_corr).to_csv(
            out_dir / f"may_training_efficiency_correlations_{self.name}.csv", index=False)
        agg_eff = (
            edf.groupby(["condition", "complexity"])["epochs_to_max_acc"]
            .agg(epochs_to_max_median="median", epochs_to_max_std="std", epochs_to_max_n="count")
            .reset_index()
        )
        agg_eff.to_csv(
            out_dir / f"may_training_efficiency_by_condition_complexity_{self.name}.csv", index=False)
        agg_eff_h = (
            edf.groupby(["condition", "hidden_size", "complexity"])["epochs_to_max_acc"]
            .agg(epochs_to_max_median="median", epochs_to_max_std="std", epochs_to_max_n="count")
            .reset_index()
        )
        agg_eff_h.to_csv(
            out_dir / f"may_training_efficiency_by_condition_hidden_complexity_{self.name}.csv",
            index=False)

        h48_vals = sorted(edf["hidden_size"].dropna().unique())
        h48_target = 48 if 48 in h48_vals else (h48_vals[0] if h48_vals else None)
        if h48_target is not None:
            edf48 = edf[edf["hidden_size"] == h48_target].copy()
            unary_x = sorted(
                edf48[edf48["condition"].isin(["n+neg", "n-neg"])]["complexity"]
                .dropna().unique()
            )

            def _snap_log(c, refs):
                if not refs:
                    return c
                log_c = math.log(float(c) + 1e-9)
                return min(refs, key=lambda u: abs(math.log(float(u) + 1e-9) - log_c))

            edf48["plot_x"] = edf48.apply(
                lambda row: _snap_log(row["complexity"], unary_x)
                if row["condition"] == "n-n" else row["complexity"],
                axis=1,
            )
            agg48 = (
                edf48.groupby(["condition", "plot_x"])["epochs_to_max_acc"]
                .agg(
                    median="median",
                    q25=lambda x: np.nanquantile(x, 0.25),
                    q75=lambda x: np.nanquantile(x, 0.75),
                )
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(10, 5.5))
            all48_x = sorted(agg48["plot_x"].dropna().unique())
            for cond in cond_order:
                g = agg48[agg48["condition"] == cond].sort_values("plot_x")
                if g.empty:
                    continue
                color = cond_colors[cond]
                linestyle = "--" if cond == "n-n" else "-"
                ax.fill_between(g["plot_x"], g["q25"], g["q75"], alpha=0.13, color=color)
                lbl = cond_labels.get(cond, cond)
                if cond == "n-n":
                    lbl += "  (x-axis snapped to nearest unary)"
                ax.plot(g["plot_x"], g["median"], marker="o", linewidth=2.2, markersize=6,
                        color=color, linestyle=linestyle, label=lbl)
            _set_log2_xticks(ax, all48_x)
            ax.set_xlabel("predicate space size (real count)", fontsize=10)
            ax.set_ylabel("epochs to max accuracy (median)", fontsize=10)
            ax.set_title(f"Training efficiency by condition  [hidden size = {int(h48_target)}]",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"may_training_efficiency_h{int(h48_target)}_{self.name}.png",
                        dpi=150)
            plt.close(fig)

        # ----------------------------------------------------------------
        # 4. TRAINING EFFICIENCY OVERVIEW — two grouped tables
        # ----------------------------------------------------------------
        _progress(5, _steps[4])

        def _v1_col(complexity, condition):
            """v1 column: M (num_predicates) for n+neg/n-neg; extracted n for n-n."""
            c = int(complexity)
            if condition == "n-n":
                return int(round(-1.0 + math.sqrt(1.0 + c)))
            return c  # n-neg: c = n;  n+neg: c = 2n  — both are already M

        def _v2_col(complexity, condition):
            """v2 column: M for n+neg/n-neg; n² for n-n (matches topsim x-axis)."""
            c = int(complexity)
            if condition == "n-n":
                n = int(round(-1.0 + math.sqrt(1.0 + c)))
                return n * n
            return c

        agg_v = agg_eff_h.copy()
        agg_v["n_p"]   = agg_v.apply(lambda r: _v1_col(r["complexity"], r["condition"]), axis=1)
        agg_v["real_c"] = agg_v.apply(lambda r: _v2_col(r["complexity"], r["condition"]), axis=1)

        v1_cols = sorted(agg_v["n_p"].dropna().unique().astype(int))
        v2_cols = sorted(agg_v["real_c"].dropna().unique().astype(int))

        def _build_eff_grouped(col_key, col_vals):
            rows = []
            for cond in cond_order:
                for h in hidden_vals:
                    sub = agg_v[(agg_v["condition"] == cond) &
                                (agg_v["hidden_size"] == h)]
                    row = {"condition": cond, "hidden": int(h)}
                    for cv in col_vals:
                        match = sub[sub[col_key] == cv]
                        row[cv] = (round(float(match["epochs_to_max_median"].iloc[0]), 1)
                                   if not match.empty else np.nan)
                    rows.append(row)
            return pd.DataFrame(rows)

        df_v1 = _build_eff_grouped("n_p", v1_cols)
        df_v2 = _build_eff_grouped("real_c", v2_cols)

        cond_short = {"n+neg": "$n^+$", "n-neg": "$n^-$", "n-n": "$n$-$n$"}

        def _write_eff_grouped_tex(path, df_tbl, col_vals, col_key_lbl, caption, label):
            def _cv(v):
                if isinstance(v, float) and np.isnan(v):
                    return r"\textemdash"
                if isinstance(v, float):
                    return f"{v:.1f}"
                return str(v)
            n_c = len(col_vals)
            col_fmt = "ll" + "r" * n_c
            hdr = (r"\textbf{Cond.} & \textbf{Hidden} & "
                   + " & ".join(rf"\textbf{{{int(cv)}}}" for cv in col_vals)
                   + r" \\")
            lines = [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                f"\\begin{{tabular}}{{{col_fmt}}}",
                r"\toprule",
                hdr,
                r"\midrule",
            ]
            for cond in cond_order:
                cond_rows = df_tbl[df_tbl["condition"] == cond]
                n_sub = len(cond_rows)
                lbl = cond_short.get(cond, cond)
                for k, (_, row) in enumerate(cond_rows.iterrows()):
                    cond_cell = (f"\\multirow{{{n_sub}}}{{*}}{{{lbl}}}"
                                 if k == 0 else "")
                    vals_str = " & ".join(_cv(row[cv]) for cv in col_vals)
                    lines.append(f"{cond_cell} & {int(row['hidden'])} & {vals_str} \\\\")
                lines.append(r"\midrule")
            lines[-1] = r"\bottomrule"
            lines += [
                r"\end{tabular}",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                r"\end{table}",
                "",
            ]
            with open(path, "w") as fh:
                fh.write("\n".join(lines))

        df_v1.to_csv(
            out_dir / f"may_training_efficiency_table_v1_{self.name}.csv", index=False)
        _write_eff_grouped_tex(
            out_dir / f"may_training_efficiency_table_v1_{self.name}.tex",
            df_v1, v1_cols, r"$n$ (hparams)",
            caption=(
                r"Median epochs to maximum accuracy. "
                r"Rows: condition $\times$ hidden size. "
                r"Columns: real message-space size $M$ "
                r"($n$ for $n^-$, $2n$ for $n^+$, base $n$ extracted from $n^2+2n$ for $n$-$n$). "
                r"Dashes indicate no data."
            ),
            label="tab:training_efficiency_v1",
        )

        df_v2.to_csv(
            out_dir / f"may_training_efficiency_table_v2_{self.name}.csv", index=False)
        _write_eff_grouped_tex(
            out_dir / f"may_training_efficiency_table_v2_{self.name}.tex",
            df_v2, v2_cols, r"predicate count",
            caption=(
                r"Median epochs to maximum accuracy. "
                r"Columns: $M$ for $n^-$ and $n^+$ (same as v1), "
                r"$n^2$ for $n$-$n$ (matching topsim x-axis). "
                r"Alignment: $n^-$ at $M=n$ and $n^+$ at $M=2n$ share a column "
                r"when they have the same real predicate count."
            ),
            label="tab:training_efficiency_v2",
        )

        # ----------------------------------------------------------------
        # 5. NEGATION HELPS LEARNING — delta epochs (+neg − −neg)
        # ----------------------------------------------------------------
        _progress(6, _steps[5])
        unary_edf = edf[edf["condition"].isin(["n+neg", "n-neg"])].copy()
        help_rows = []
        sig_rows = []
        for h in sorted(pd.to_numeric(unary_edf["hidden_size"], errors="coerce").dropna().unique()):
            for c in sorted(pd.to_numeric(unary_edf["complexity"], errors="coerce").dropna().unique()):
                sub = unary_edf[(unary_edf["hidden_size"] == h) & (unary_edf["complexity"] == c)]
                a = sub.loc[sub["condition"] == "n+neg", "epochs_to_max_acc"].to_numpy(dtype=float)
                b = sub.loc[sub["condition"] == "n-neg", "epochs_to_max_acc"].to_numpy(dtype=float)
                if len(a) == 0 and len(b) == 0:
                    continue
                median_a = float(np.nanmedian(a)) if len(a) else np.nan
                median_b = float(np.nanmedian(b)) if len(b) else np.nan
                std_a = float(np.nanstd(a, ddof=1)) if np.isfinite(a).sum() >= 2 else np.nan
                std_b = float(np.nanstd(b, ddof=1)) if np.isfinite(b).sum() >= 2 else np.nan
                help_rows.append({
                    "hidden_size": h, "complexity": c,
                    "epochs_plus_neg_median": median_a, "epochs_plus_neg_std": std_a,
                    "epochs_minus_neg_median": median_b, "epochs_minus_neg_std": std_b,
                    "delta_epochs_plus_minus": (median_a - median_b)
                    if np.isfinite(median_a) and np.isfinite(median_b) else np.nan,
                })
                sig_rows.append({
                    "hidden_size": h, "complexity": c,
                    "n_plus_neg": int(np.isfinite(a).sum()),
                    "n_minus_neg": int(np.isfinite(b).sum()),
                    "p_welch": self._ttest_p(a, b),
                })
        help_df = pd.DataFrame(help_rows)
        sig_df = pd.DataFrame(sig_rows)
        help_df.to_csv(out_dir / f"may_negation_helps_learning_delta_{self.name}.csv", index=False)
        sig_df.to_csv(
            out_dir / f"may_negation_helps_learning_significance_{self.name}.csv", index=False)
        if not help_df.empty:
            # Plot: hidden=48 only (single curve)
            plot_h = h48_target if h48_target in help_df["hidden_size"].values else \
                     help_df["hidden_size"].iloc[0]
            h48_help = help_df[help_df["hidden_size"] == plot_h].sort_values("complexity")
            if not h48_help.empty:
                fig, ax = plt.subplots(figsize=(9, 5))
                g = h48_help
                ticks = sorted(g["complexity"].dropna().unique())
                # error band: propagate std of each arm
                combined_std = (g["epochs_plus_neg_std"].fillna(0)
                                + g["epochs_minus_neg_std"].fillna(0))
                ax.fill_between(g["complexity"],
                                g["delta_epochs_plus_minus"] - combined_std,
                                g["delta_epochs_plus_minus"] + combined_std,
                                alpha=0.13, color="#1f77b4")
                ax.plot(g["complexity"], g["delta_epochs_plus_minus"],
                        marker="o", linewidth=2.2, markersize=6, color="#1f77b4")
                ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.5,
                           label="no difference")
                ax.set_xscale("log", base=2)
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [str(int(t)) if float(t).is_integer() else f"{t:g}" for t in ticks],
                    rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("predicate space size", fontsize=10)
                ax.set_ylabel(r"$\Delta$ epochs  (+neg $-$ $-$neg)", fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(out_dir / f"may_negation_helps_learning_{self.name}.png", dpi=150)
                plt.close(fig)

                # LaTeX TikZ export — written directly for full axis control
                # Drop rows where delta is undefined (one arm missing)
                g_tex = g.dropna(subset=["delta_epochs_plus_minus"])
                delta_vals = g_tex["delta_epochs_plus_minus"].to_numpy(dtype=float)
                combined_std_vals = (g_tex["epochs_plus_neg_std"].fillna(0)
                                     + g_tex["epochs_minus_neg_std"].fillna(0)).to_numpy()
                cx = g_tex["complexity"].to_numpy()
                pad = max(2.0, (delta_vals.max() - delta_vals.min()) * 0.25)
                y_lo = float(delta_vals.min() - pad)
                y_hi = float(delta_vals.max() + pad)
                x_min_t = int(cx.min()); x_max_t = int(cx.max())

                # significance for plot_h from sig_df
                sig_h = sig_df[sig_df["hidden_size"] == plot_h].set_index("complexity")

                def _flt(v):
                    v = float(v)
                    if not np.isfinite(v):
                        return "0"
                    return str(int(v)) if v == int(v) else f"{v:.4g}"

                nl_lines = [
                    r"\begin{figure}[t]",
                    r"\centering",
                    r"\begin{tikzpicture}",
                    r"\begin{axis}[",
                    r"    width=\linewidth,",
                    r"    height=0.55\linewidth,",
                    r"    grid=both,",
                    r"    legend style={at={(0.5,1.02)},anchor=south,legend columns=2},",
                    r"    xlabel={predicate space size},",
                    r"    ylabel={$\Delta$ epochs ($+$neg $-$ $-$neg)},",
                    r"    xmode=log, log basis x=2,",
                    f"    xtick={{{','.join(str(int(t)) for t in ticks)}}},",
                    f"    xticklabels={{{','.join(str(int(t)) for t in ticks)}}},",
                    r"    x tick label style={rotate=45,anchor=east},",
                    f"    ymin={y_lo:.2f}, ymax={y_hi:.2f},",
                    f"    xmin={x_min_t}, xmax={x_max_t},",
                    r"]",
                    "",
                    r"% shaded std band",
                    r"\addplot[fill=blue!13, draw=none, forget plot] coordinates {",
                ]
                for xv, dv, sv in zip(cx, delta_vals, combined_std_vals):
                    nl_lines.append(f"    ({_flt(xv)},{_flt(dv + sv)})")
                for xv, dv, sv in zip(reversed(cx), reversed(delta_vals),
                                       reversed(combined_std_vals)):
                    nl_lines.append(f"    ({_flt(xv)},{_flt(dv - sv)})")
                nl_lines += ["} -- cycle;", ""]

                nl_lines += [
                    r"% delta curve",
                    r"\addplot[mark=*,thick,color=blue!70!black] coordinates {",
                ]
                for xv, dv in zip(cx, delta_vals):
                    nl_lines.append(f"    ({_flt(xv)},{_flt(dv)})")
                nl_lines += ["};", ""]

                nl_lines += [
                    r"% y=0 reference",
                    f"\\addplot[dashed,gray,thick,forget plot] coordinates "
                    f"{{({x_min_t},0) ({x_max_t},0)}};",
                    r"\addlegendentry{no difference}",
                    "",
                ]

                # annotate significant points with * above the marker
                for xv, dv in zip(cx, delta_vals):
                    row_sig = (sig_h.loc[xv] if xv in sig_h.index else None)
                    if row_sig is not None:
                        p_val = float(row_sig.get("p_welch", float("nan"))
                                      if hasattr(row_sig, "get")
                                      else row_sig["p_welch"])
                        if np.isfinite(p_val) and p_val < 0.05:
                            nl_lines.append(
                                f"\\node[above,font=\\small\\bfseries,text=red] at "
                                f"(axis cs:{_flt(xv)},{_flt(dv)}) {{*}};")

                nl_lines += [
                    r"\end{axis}",
                    r"\end{tikzpicture}",
                    (r"\caption{Difference in median epochs to maximum accuracy between "
                     r"games with and without negation (hidden size $="
                     + str(int(plot_h))
                     + r"$). Negative values indicate negation accelerates convergence. "
                     r"Shaded band: combined standard deviation. "
                     r"Asterisk (*): $p < 0.05$ (Welch $t$-test).}"),
                    r"\label{fig:negation_helps_learning}",
                    r"\end{figure}",
                    "",
                ]
                with open(out_dir / f"may_negation_helps_learning_{self.name}.tex",
                          "w") as fh:
                    fh.write("\n".join(nl_lines))

                # Significance table (all hidden sizes)
                sig_tbl = sig_df[sig_df["n_plus_neg"] > 0].copy()
                sig_tbl = sig_tbl[sig_tbl["n_minus_neg"] > 0].copy()
                sig_tbl["p_welch"] = sig_tbl["p_welch"].apply(
                    lambda v: f"{v:.3f}" if np.isfinite(v) else "---")
                sig_pivot = sig_tbl.pivot_table(
                    index="complexity", columns="hidden_size",
                    values="p_welch", aggfunc="first"
                ).reset_index()
                sig_pivot.columns.name = None
                h_cols = [c for c in sig_pivot.columns if c != "complexity"]
                sig_pivot.rename(columns={"complexity": "pred. space"}, inplace=True)
                sig_pivot.to_csv(
                    out_dir / f"may_negation_helps_learning_sig_table_{self.name}.csv",
                    index=False)
                self._write_latex_table(
                    out_dir / f"may_negation_helps_learning_sig_table_{self.name}.tex",
                    sig_pivot,
                    col_headers=(
                        ["pred. space"]
                        + [rf"$p$ (h={int(h)})" for h in h_cols]
                    ),
                    caption=(
                        r"Welch $t$-test $p$-values for the difference in epochs to "
                        r"maximum accuracy between games with and without negation, "
                        r"per predicate space size and hidden size. "
                        r"Bold: $p < 0.05$."
                    ),
                    label="tab:negation_helps_significance",
                )

        # ----------------------------------------------------------------
        # 6. MESSAGE EFFICIENCY — snapshot at best epoch, by negation
        # ----------------------------------------------------------------
        _progress(7, _steps[6])
        mdf = df[df["condition"].isin(["n+neg", "n-neg"])].dropna(
            subset=["msg_eff", "complexity"]).copy()
        msg_sig_rows = []
        for c in sorted(pd.to_numeric(mdf["complexity"], errors="coerce").dropna().unique()):
            sub = mdf[mdf["complexity"] == c]
            a = sub.loc[sub["condition"] == "n+neg", "msg_eff"].to_numpy(dtype=float)
            b = sub.loc[sub["condition"] == "n-neg", "msg_eff"].to_numpy(dtype=float)
            msg_sig_rows.append({
                "complexity": c,
                "n_plus_neg": int(np.isfinite(a).sum()),
                "n_minus_neg": int(np.isfinite(b).sum()),
                "p_welch": self._ttest_p(a, b),
            })
        pd.DataFrame(msg_sig_rows).to_csv(
            out_dir / f"may_message_efficiency_significance_{self.name}.csv", index=False)
        mdf.sort_values(["condition", "hidden_size", "complexity", "run_name"]).to_csv(
            out_dir / f"may_message_efficiency_points_{self.name}.csv", index=False)

        if not mdf.empty:
            overall_p = self._ttest_p(
                mdf.loc[mdf["condition"] == "n+neg", "msg_eff"].to_numpy(dtype=float),
                mdf.loc[mdf["condition"] == "n-neg", "msg_eff"].to_numpy(dtype=float),
            )
            msg_agg = (
                mdf.groupby(["condition", "complexity"])["msg_eff"]
                .agg(msg_eff_median="median", msg_eff_std="std", msg_eff_n="count")
                .reset_index()
            )
            msg_agg.to_csv(
                out_dir / f"may_message_efficiency_by_condition_complexity_{self.name}.csv",
                index=False)
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for cond, color in [("n+neg", "#1f77b4"), ("n-neg", "#e07a2d")]:
                g = msg_agg[msg_agg["condition"] == cond].sort_values("complexity")
                if g.empty:
                    continue
                lo = (g["msg_eff_median"] - g["msg_eff_std"].fillna(0)).clip(lower=0)
                hi = g["msg_eff_median"] + g["msg_eff_std"].fillna(0)
                ax.fill_between(g["complexity"], lo, hi, alpha=0.13, color=color)
                ax.plot(g["complexity"], g["msg_eff_median"], marker="o", linewidth=2.2,
                        markersize=6, color=color, label=cond_labels.get(cond, cond))
            ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.3, label="optimal (1.0)")
            ticks = sorted(msg_agg["complexity"].dropna().unique())
            _set_log2_xticks(ax, ticks)
            ax.set_xlabel("predicate space size", fontsize=10)
            ax.set_ylabel("inverse compression ratio", fontsize=10)
            ax.set_title(
                f"Message efficiency by negation  ({self._format_p_value(overall_p)} overall)",
                fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"may_message_efficiency_by_negation_{self.name}.png", dpi=150)
            plt.close(fig)

    def _prepare_negation_top_df(self, neg_df):
        if neg_df is None or len(neg_df) == 0:
            return None
        df = neg_df.copy()
        # Backward/forward compatibility for metric naming:
        # old: l_X, l_T, r ; new: h, t, c
        if "h" in df.columns and "l_X" not in df.columns:
            df["l_X"] = df["h"]
        if "t" in df.columns and "l_T" not in df.columns:
            df["l_T"] = df["t"]
        if "c" in df.columns and "r" not in df.columns:
            df["r"] = df["c"]

        req = {"run_name", "n", "l_X", "l_T", "r", "atoms", "support"}
        if not req.issubset(set(df.columns)):
            print(f"[WARN] Negation dataframe missing columns: {sorted(req - set(df.columns))}")
            return None

        df["run_name"] = df["run_name"].astype(str)
        meta = self._build_run_meta_df()[["run_name", "complexity", "hidden_size", "no_negation", "no_conjunction", "properties", "reaper_interval", "voc_penalty"]]
        needed_meta = [c for c in ["complexity", "hidden_size", "no_negation", "no_conjunction", "reaper_interval", "voc_penalty"] if c not in df.columns]
        if needed_meta:
            df = df.merge(meta[["run_name"] + needed_meta], on="run_name", how="left")
        if "properties" not in df.columns:
            df = df.merge(meta[["run_name", "properties"]], on="run_name", how="left")
        if "no_negation" in df.columns:
            df = df[df["no_negation"] == False]
        for c in ["complexity", "hidden_size", "reaper_interval", "voc_penalty", "n", "l_X", "l_T", "r", "atoms", "support"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "properties" in df.columns:
            df["properties"] = df["properties"].astype(str)
        df["leak_unary_x"] = np.where(df["atoms"] <= 1, df["l_X"], np.nan)
        df["leak_x"] = df["l_X"]
        df["leak_composite_t"] = np.where(df["atoms"] > 1, df["l_T"], np.nan)
        return df.dropna(subset=["complexity", "n"])

    def plot_negation_metrics_by_voc_penalty(self, neg_df, *, out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return
        if "voc_penalty" not in df.columns:
            print("[INFO] voc_penalty plot skipped (missing voc_penalty metadata).")
            return
        df = df[df["voc_penalty"].notna()]
        if df.empty:
            print("[INFO] voc_penalty plot skipped (no valid voc_penalty values).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metric_defs = [
            ("n", "n", "#1f77b4"),
            ("l_X", "l_X", "#ff7f0e"),
            ("leak_composite_t", "l_T (composite)", "#2ca02c"),
            ("r", "r", "#d62728"),
        ]

        def _format_penalty_tick(x):
            if not np.isfinite(x):
                return ""
            if abs(float(x)) < 1e-15:
                return "0"
            s = f"{float(x):.12f}".rstrip("0").rstrip(".")
            return s if s else "0"

        fig = plt.figure(figsize=(10, 6))
        plotted = 0
        for col, label, color in metric_defs:
            sub = df[df[col].notna()]
            if sub.empty:
                continue
            stats = sub.groupby("voc_penalty")[col].agg(mean="mean", median="median").reset_index().sort_values("voc_penalty")
            plt.plot(stats["voc_penalty"], stats["mean"], marker="o", color=color, linewidth=2.0, label=f"{label} mean")
            plt.plot(stats["voc_penalty"], stats["median"], marker="o", linestyle=":", color=color, alpha=0.9, label=f"{label} median")
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            return

        xticks = sorted(pd.to_numeric(df["voc_penalty"], errors="coerce").dropna().unique())
        pos_ticks = [x for x in xticks if x > 0]
        if len(pos_ticks) > 0:
            # Log-like spacing for positive penalties; keeps 0 visible if present.
            if any(x <= 0 for x in xticks):
                linthresh = min(pos_ticks) / 2.0
                plt.xscale("symlog", linthresh=linthresh, base=10)
            else:
                plt.xscale("log", base=10)
        if xticks:
            if len(xticks) > 10:
                idx = np.unique(np.linspace(0, len(xticks) - 1, num=10).round().astype(int))
                shown_ticks = [xticks[i] for i in idx]
            else:
                shown_ticks = xticks
            labels = [_format_penalty_tick(x) for x in shown_ticks]
            rot = 0 if len(xticks) <= 8 else 45
            plt.xticks(shown_ticks, labels, rotation=rot, ha="right" if rot else "center")
        plt.title("negation by voc_penalty")
        plt.xlabel("voc_penalty")
        plt.ylabel("score")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"negation_by_voc_penalty_{self.name}.png", dpi=150)
        plt.close(fig)

    def plot_f_scores_by_voc_penalty(self, neg_df, *, out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return
        if "voc_penalty" not in df.columns:
            print("[INFO] voc_penalty F-score plot skipped (missing voc_penalty metadata).")
            return
        df = df[df["voc_penalty"].notna()]
        if df.empty:
            print("[INFO] voc_penalty F-score plot skipped (no valid voc_penalty values).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metric_defs = [
            ("F1_nr", "F1_nr", "#9467bd"),
            ("F1_nT", "F1_nT", "#8c564b"),
        ]

        def _format_penalty_tick(x):
            if not np.isfinite(x):
                return ""
            if abs(float(x)) < 1e-15:
                return "0"
            s = f"{float(x):.12f}".rstrip("0").rstrip(".")
            return s if s else "0"

        fig = plt.figure(figsize=(10, 6))
        plotted = 0
        for col, label, color in metric_defs:
            if col not in df.columns:
                continue
            sub = df[df[col].notna()]
            if sub.empty:
                continue
            stats = sub.groupby("voc_penalty")[col].agg(mean="mean", median="median").reset_index().sort_values("voc_penalty")
            plt.plot(stats["voc_penalty"], stats["mean"], marker="o", color=color, linewidth=2.0, label=f"{label} mean")
            plt.plot(stats["voc_penalty"], stats["median"], marker="o", linestyle=":", color=color, alpha=0.9, label=f"{label} median")
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            return

        xticks = sorted(pd.to_numeric(df["voc_penalty"], errors="coerce").dropna().unique())
        pos_ticks = [x for x in xticks if x > 0]
        if len(pos_ticks) > 0:
            if any(x <= 0 for x in xticks):
                linthresh = min(pos_ticks) / 2.0
                plt.xscale("symlog", linthresh=linthresh, base=10)
            else:
                plt.xscale("log", base=10)
        if xticks:
            if len(xticks) > 10:
                idx = np.unique(np.linspace(0, len(xticks) - 1, num=10).round().astype(int))
                shown_ticks = [xticks[i] for i in idx]
            else:
                shown_ticks = xticks
            labels = [_format_penalty_tick(x) for x in shown_ticks]
            rot = 0 if len(xticks) <= 8 else 45
            plt.xticks(shown_ticks, labels, rotation=rot, ha="right" if rot else "center")
        plt.title("negation f-scores by voc_penalty")
        plt.xlabel("voc_penalty")
        plt.ylabel("score")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"negation_f_scores_by_voc_penalty_{self.name}.png", dpi=150)
        plt.close(fig)

    def plot_negation_metrics_by_reaper_interval(self, neg_df, *, profile_tag="", out_dir="plots", metrics=None):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return
        if "reaper_interval" not in df.columns:
            return

        df = df.copy()
        df = df[df["reaper_interval"].notna()]
        df = df[df["reaper_interval"] > 0]
        if df.empty:
            print("[INFO] Reaper-interval plots skipped (no valid reaper interval in config).")
            return

        # F1 between negation score and non-leak composite score.
        non_leak_t = 1.0 - df["leak_composite_t"]
        denom = df["n"] + non_leak_t
        df["f1_nT"] = np.where((denom > 0) & np.isfinite(non_leak_t), 2.0 * df["n"] * non_leak_t / denom, np.nan)

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = str(profile_tag) if profile_tag else "default"

        complexities = sorted(pd.to_numeric(df["complexity"], errors="coerce").dropna().unique())
        if not complexities:
            return
        palette = sns.color_palette("viridis", n_colors=max(1, len(complexities)))
        c_map = {c: palette[i] for i, c in enumerate(complexities)}
        c_label = {}
        for c in complexities:
            props = df.loc[df["complexity"] == c, "properties"].dropna().astype(str).unique()
            props = sorted(props, key=self._properties_sort_key)
            c_label[c] = props[0] if len(props) else str(int(c))

        if metrics is None:
            metrics = ["f1_nT", "n", "l_X", "leak_composite_t", "r"]

        metric_spec = {
            "f1_nT": ("F1_nT", "f1_nt"),
            "n": ("n", "n"),
            "l_X": ("l_X", "l_x"),
            "leak_composite_t": ("l_T (composite)", "l_t_composite"),
            "r": ("r", "r"),
        }

        def _clean_interval_ticks(values):
            ticks = sorted(pd.to_numeric(values, errors="coerce").dropna().unique())
            if len(ticks) <= 10:
                return ticks
            # Keep ticks legible when many intervals are present.
            idx = np.unique(np.linspace(0, len(ticks) - 1, num=10).round().astype(int))
            return [ticks[i] for i in idx]

        def _plot_interval_metric(metric_col, metric_label, file_suffix):
            sub = df[df[metric_col].notna()]
            if sub.empty:
                return
            interval_ticks = _clean_interval_ticks(sub["reaper_interval"])
            fig = plt.figure(figsize=(10, 6))
            for c in complexities:
                g = sub[sub["complexity"] == c]
                if g.empty:
                    continue
                stats = (
                    g.groupby("reaper_interval")[metric_col]
                    .agg(mean="mean")
                    .reset_index()
                    .sort_values("reaper_interval")
                )
                col = c_map[c]
                plt.plot(stats["reaper_interval"], stats["mean"], marker="o", linewidth=2.0, color=col, label=f"{c_label[c]} mean")

            plt.xscale("log")
            if interval_ticks:
                labels = [str(int(x)) if float(x).is_integer() else f"{x:g}" for x in interval_ticks]
                rot = 0 if len(interval_ticks) <= 8 else 45
                plt.xticks(interval_ticks, labels, rotation=rot, ha="right" if rot else "center")
            plt.xlabel("reaper interval")
            plt.ylabel(metric_label)
            plt.title(f"{metric_label} by reaper interval ({tag})")
            plt.grid(True, alpha=0.3)
            plt.legend(ncol=2, title="properties")
            fig.tight_layout()
            fig.savefig(out_dir / f"negation_{file_suffix}_by_reaper_interval_{tag}_{self.name}.png", dpi=150)
            plt.close(fig)

        for metric in metrics:
            spec = metric_spec.get(metric)
            if spec is None:
                continue
            label, suffix = spec
            _plot_interval_metric(metric, label, suffix)

    def plot_reaper_step_vs_f_scores(self, neg_df, *, profile_tag="", out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return
        if "reaper_interval" not in df.columns:
            return

        df = df.copy()
        df = df[df["reaper_interval"].notna()]
        df = df[df["reaper_interval"] > 0]
        if df.empty:
            return

        if "F1_nT" not in df.columns:
            n = pd.to_numeric(df.get("n"), errors="coerce")
            l_t = pd.to_numeric(df.get("l_T"), errors="coerce")
            den = n + l_t
            df["F1_nT"] = np.where((den > 0) & n.notna() & l_t.notna(), 2.0 * n * l_t / den, np.nan)
        if "F1_nr" not in df.columns:
            n = pd.to_numeric(df.get("n"), errors="coerce")
            r = pd.to_numeric(df.get("r"), errors="coerce")
            den = n + r
            df["F1_nr"] = np.where((den > 0) & n.notna() & r.notna(), 2.0 * n * r / den, np.nan)

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = str(profile_tag) if profile_tag else "default"

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        x_raw = pd.to_numeric(df["reaper_interval"], errors="coerce").to_numpy(dtype=float)
        x = np.log2(np.maximum(x_raw, 1.0))
        z = np.log2(np.maximum(pd.to_numeric(df["complexity"], errors="coerce").to_numpy(dtype=float), 1.0))
        metric_defs = [("F1_nT", "F1_nT"), ("F1_nr", "F1_nr")]

        for ax, (col, label) in zip(axes, metric_defs):
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_raw)
            if mask.sum() < 2:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                continue

            sc = ax.scatter(x_raw[mask], y[mask], c=z[mask], cmap="viridis", alpha=0.85, edgecolors="none")
            ax.set_xscale("log", base=2)

            if np.ptp(x[mask]) > 1e-12:
                a, b = np.polyfit(x[mask], y[mask], deg=1)
                xg_raw = np.logspace(np.log2(x_raw[mask].min()), np.log2(x_raw[mask].max()), 120, base=2.0)
                xg = np.log2(xg_raw)
                ax.plot(xg_raw, a * xg + b, color="#111111", linewidth=2.0)

            r_val, p_val = self._pearson_r_p(x[mask], y[mask])
            r_txt = f"r={r_val:.3f}" if np.isfinite(r_val) else "r=nan"
            ax.set_title(f"{label} vs reaper_step ({r_txt}, {self._format_p_value(p_val)})")
            ax.set_xlabel("reaper_step")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("log2(complexity)")

        fig.suptitle(f"reaper step correlation with F-scores ({tag})")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"reaper_step_vs_f_scores_{tag}_{self.name}.png", dpi=150)
        plt.close(fig)

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

    def _leak_metric_specs(self):
        return [
            ("leak_unary_x", "l_X (unary)", "leak_unary_x", "#2ca02c"),
            ("leak_x", "l_X (all)", "leak_x", "#ff7f0e"),
            ("leak_composite_t", "l_T (composite)", "leak_composite_t", "#1f77b4"),
        ]

    def _pearson_r_p(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return np.nan, np.nan
        x_m = x[mask]
        y_m = y[mask]
        if np.ptp(x_m) <= 1e-12 or np.ptp(y_m) <= 1e-12:
            return np.nan, np.nan
        r, p = pearsonr(x_m, y_m)
        return float(r), float(p)

    def _spearman_r_p(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return np.nan, np.nan
        r, p = spearmanr(x[mask], y[mask])
        return float(r), float(p)

    def _format_p_value(self, p):
        if not np.isfinite(p):
            return "p=nan"
        if p < 1e-3:
            return f"p={p:.1e}"
        return f"p={p:.3f}"

    def _negation_timeline_df(self, neg_df):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return None
        if "epoch_analyzed" not in df.columns:
            print("[WARN] Negation dataframe has no epoch_analyzed column; timeline plots skipped.")
            return None

        df = df.copy()
        df["epoch_analyzed"] = pd.to_numeric(df["epoch_analyzed"], errors="coerce")
        df = df.dropna(subset=["epoch_analyzed"])
        if df.empty:
            return None

        order_cols = ["run_name", "epoch_analyzed", "n", "r", "l_T", "l_X"]
        order_asc = [True, True, False, False, True, True]
        ranked = df.sort_values(order_cols, ascending=order_asc, na_position="last")
        key = ["run_name", "epoch_analyzed"]

        best = ranked.groupby(key, as_index=False).first()
        comp = ranked[ranked["atoms"] > 1]
        if comp.empty:
            best["l_T_composite"] = np.nan
        else:
            comp_best = comp.groupby(key, as_index=False).first()[key + ["l_T"]]
            comp_best = comp_best.rename(columns={"l_T": "l_T_composite"})
            best = best.merge(comp_best, on=key, how="left")

        return best

    def _plot_timeline_metric_set(self, timeline_df, metrics, title, filename, out_dir):
        if timeline_df is None or timeline_df.empty:
            return
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        complexities = sorted(pd.to_numeric(timeline_df["complexity"], errors="coerce").dropna().unique())
        if not complexities:
            return
        prop_label = {}
        for c in complexities:
            props = (
                timeline_df.loc[timeline_df["complexity"] == c, "properties"]
                .dropna()
                .astype(str)
                .unique()
            )
            props = sorted(props, key=self._properties_sort_key)
            if len(props) == 0:
                prop_label[c] = str(int(c))
            elif len(props) == 1:
                prop_label[c] = props[0]
            else:
                prop_label[c] = "/".join(props)
        palette = sns.color_palette("viridis", n_colors=max(1, len(complexities)))
        c_map = {c: palette[i] for i, c in enumerate(complexities)}

        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, metrics):
            sub_all = timeline_df[timeline_df[col].notna()]
            for c in complexities:
                sub = sub_all[sub_all["complexity"] == c]
                if sub.empty:
                    continue
                stats = (
                    sub.groupby("epoch_analyzed")[col]
                    .agg(
                        median="median",
                        q25=lambda x: np.nanquantile(x, 0.25),
                        q75=lambda x: np.nanquantile(x, 0.75),
                    )
                    .reset_index()
                    .sort_values("epoch_analyzed")
                )
                colr = c_map[c]
                ax.fill_between(stats["epoch_analyzed"], stats["q25"], stats["q75"], color=colr, alpha=0.15)
                ax.plot(stats["epoch_analyzed"], stats["median"], color=colr, linewidth=2.0, label=prop_label[c])
            ax.set_title(label)
            ax.set_ylabel("score")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("epoch")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle(title, y=0.995)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=min(6, len(labels)),
                frameon=False,
                title="properties",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.90])
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)

    def plot_negation_metrics_over_epochs(self, neg_df, *, profile_tag="", out_dir="plots"):
        timeline_df = self._negation_timeline_df(neg_df)
        if timeline_df is None or timeline_df.empty:
            return
        if timeline_df["epoch_analyzed"].nunique() < 2:
            print("[INFO] Negation timeline plots skipped (only one epoch available).")
            return

        tag = str(profile_tag) if profile_tag else "default"
        self._plot_timeline_metric_set(
            timeline_df,
            metrics=[("n", "n"), ("l_X", "l_X"), ("l_T_composite", "l_T (composite)")],
            title=f"negation metrics over epochs ({tag})",
            filename=f"negation_timeline_n_lx_ltcomp_{tag}_{self.name}.png",
            out_dir=out_dir,
        )
        self._plot_timeline_metric_set(
            timeline_df,
            metrics=[("n", "n"), ("r", "r")],
            title=f"negation and remainder over epochs ({tag})",
            filename=f"negation_timeline_n_r_{tag}_{self.name}.png",
            out_dir=out_dir,
        )

    def plot_negation_metrics_by_complexity(self, neg_df, *, out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        reaper_vals = sorted(pd.to_numeric(df.get("reaper_interval"), errors="coerce").dropna().unique()) if "reaper_interval" in df.columns else []
        use_reaper_split = len(reaper_vals) > 1

        # combined 4-metric view across complexities
        metric_defs = [
            ("n", "n", "#1f77b4"),
            ("l_X", "l_X", "#ff7f0e"),
            ("leak_composite_t", "l_T (composite)", "#2ca02c"),
            ("r", "r", "#d62728"),
        ]
        if use_reaper_split:
            complexities = sorted(pd.to_numeric(df["complexity"], errors="coerce").dropna().unique())
            if not complexities:
                return
            pal = sns.color_palette("viridis", n_colors=max(1, len(complexities)))
            comp_color = {c: pal[i] for i, c in enumerate(complexities)}
            comp_label = {}
            for c in complexities:
                props = df.loc[df["complexity"] == c, "properties"].dropna().astype(str).unique()
                props = sorted(props, key=self._properties_sort_key)
                comp_label[c] = props[0] if len(props) else str(int(c))
            interval_ticks = sorted(pd.to_numeric(df["reaper_interval"], errors="coerce").dropna().unique())

            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            axes = axes.flatten()
            for ax, (col, label, _) in zip(axes, metric_defs):
                sub = df[df[col].notna() & df["reaper_interval"].notna()]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                for c in complexities:
                    g = sub[sub["complexity"] == c]
                    if g.empty:
                        continue
                    stats = g.groupby("reaper_interval")[col].agg(mean="mean").reset_index().sort_values("reaper_interval")
                    ax.plot(stats["reaper_interval"], stats["mean"], marker="o", linewidth=2.0, color=comp_color[c], label=comp_label[c])
                ax.set_xscale("log", base=2)
                ax.set_xticks(interval_ticks)
                ax.set_xticklabels([str(int(x)) if float(x).is_integer() else f"{x:g}" for x in interval_ticks])
                ax.set_title(label)
                ax.set_ylabel("score")
                ax.grid(True, alpha=0.3)
            axes[-2].set_xlabel("reaper interval")
            axes[-1].set_xlabel("reaper interval")
            handles, hlabels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, hlabels, title="properties", loc="upper center", ncol=min(6, len(hlabels)))
                fig.tight_layout(rect=[0, 0, 1, 0.93])
            else:
                fig.tight_layout()
            fig.suptitle("negation by reaper interval", y=0.995)
            fig.savefig(out_dir / f"negation_by_complexity_{self.name}.png", dpi=150)
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(10, 6))
            plotted = 0
            for col, label, color in metric_defs:
                sub = df[df[col].notna()]
                if sub.empty:
                    continue
                stats = sub.groupby("complexity")[col].agg(mean="mean", median="median").reset_index()
                plt.plot(stats["complexity"], stats["mean"], marker="o", color=color, linewidth=2.0, label=f"{label} mean")
                plt.plot(stats["complexity"], stats["median"], marker="o", linestyle=":", color=color, alpha=0.9, label=f"{label} median")
                plotted += 1
            if plotted > 0:
                xticks, labels = self._complexity_ticks(df)
                plt.xscale("log", base=2)
                plt.xticks(xticks, labels, rotation=45, ha="right")
                plt.title("negation by complexity")
                plt.xlabel("complexity")
                plt.ylabel("score")
                plt.grid(True, alpha=0.3)
                plt.legend(ncol=2)
                fig.tight_layout()
                fig.savefig(out_dir / f"negation_by_complexity_{self.name}.png", dpi=150)
            plt.close(fig)

        # F-score view across complexities
        fscore_defs = [
            ("F1_nr", "F1_nr", "#9467bd"),
            ("F1_nT", "F1_nT", "#8c564b"),
        ]
        if use_reaper_split:
            complexities = sorted(pd.to_numeric(df["complexity"], errors="coerce").dropna().unique())
            if not complexities:
                return
            pal = sns.color_palette("viridis", n_colors=max(1, len(complexities)))
            comp_color = {c: pal[i] for i, c in enumerate(complexities)}
            comp_label = {}
            for c in complexities:
                props = df.loc[df["complexity"] == c, "properties"].dropna().astype(str).unique()
                props = sorted(props, key=self._properties_sort_key)
                comp_label[c] = props[0] if len(props) else str(int(c))
            interval_ticks = sorted(pd.to_numeric(df["reaper_interval"], errors="coerce").dropna().unique())

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
            any_plot = False
            for ax, (col, label, _) in zip(axes, fscore_defs):
                if col not in df.columns:
                    ax.set_visible(False)
                    continue
                sub = df[df[col].notna() & df["reaper_interval"].notna()]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                for c in complexities:
                    g = sub[sub["complexity"] == c]
                    if g.empty:
                        continue
                    stats = g.groupby("reaper_interval")[col].agg(mean="mean").reset_index().sort_values("reaper_interval")
                    ax.plot(stats["reaper_interval"], stats["mean"], marker="o", linewidth=2.0, color=comp_color[c], label=comp_label[c])
                    any_plot = True
                ax.set_xscale("log", base=2)
                ax.set_xticks(interval_ticks)
                ax.set_xticklabels([str(int(x)) if float(x).is_integer() else f"{x:g}" for x in interval_ticks])
                ax.set_title(label)
                ax.set_xlabel("reaper interval")
                ax.set_ylabel("score")
                ax.grid(True, alpha=0.3)
            if any_plot:
                handles, hlabels = axes[0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, hlabels, title="properties", loc="upper center", ncol=min(6, len(hlabels)))
                    fig.tight_layout(rect=[0, 0, 1, 0.90])
                else:
                    fig.tight_layout()
                fig.suptitle("negation f-scores by reaper interval", y=0.995)
                fig.savefig(out_dir / f"negation_f_scores_by_complexity_{self.name}.png", dpi=150)
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(10, 6))
            plotted = 0
            for col, label, color in fscore_defs:
                if col not in df.columns:
                    continue
                sub = df[df[col].notna()]
                if sub.empty:
                    continue
                stats = sub.groupby("complexity")[col].agg(mean="mean", median="median").reset_index()
                plt.plot(stats["complexity"], stats["mean"], marker="o", color=color, linewidth=2.0, label=f"{label} mean")
                plt.plot(stats["complexity"], stats["median"], marker="o", linestyle=":", color=color, alpha=0.9, label=f"{label} median")
                plotted += 1
            if plotted > 0:
                xticks, labels = self._complexity_ticks(df)
                plt.xscale("log", base=2)
                plt.xticks(xticks, labels, rotation=45, ha="right")
                plt.title("negation f-scores by complexity")
                plt.xlabel("complexity")
                plt.ylabel("score")
                plt.grid(True, alpha=0.3)
                plt.legend(ncol=2)
                fig.tight_layout()
                fig.savefig(out_dir / f"negation_f_scores_by_complexity_{self.name}.png", dpi=150)
            plt.close(fig)

    def plot_negation_metric_slopes(self, neg_df, *, profile_tag="", out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        if df is None or df.empty:
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        df["run_id"] = df["run_name"].map(self._extract_run_id)
        tag = str(profile_tag) if profile_tag else "default"
        for col, label, suffix, color in self._leak_metric_specs():
            sub = df[df[col].notna()]
            if sub.empty:
                continue
            slope_df = sub.groupby(["run_id", "complexity"]).agg(leak=("{}".format(col), "mean")).reset_index()
            fig = plt.figure(figsize=(10, 6))
            for _, grp in slope_df.groupby("run_id"):
                grp = grp.sort_values("complexity")
                if len(grp) < 2:
                    continue
                plt.plot(grp["complexity"], grp["leak"], color="#808080", alpha=0.35, linewidth=1.0)
            agg = slope_df.groupby("complexity")["leak"].agg(mean="mean", median="median").reset_index()
            plt.plot(agg["complexity"], agg["mean"], color=color, marker="o", linewidth=2.2, label="mean")
            plt.plot(agg["complexity"], agg["median"], color=color, marker="o", linestyle=":", linewidth=2.0, label="median")
            xticks, labels = self._complexity_ticks(slope_df)
            plt.xscale("log", base=2)
            plt.xticks(xticks, labels, rotation=45, ha="right")
            plt.title(f"{label} slopes across complexity ({tag})")
            plt.xlabel("complexity")
            plt.ylabel("score")
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"negation_{suffix}_slopes_{tag}_{self.name}.png", dpi=150)
            plt.close(fig)

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
        cvals = np.log2(np.maximum(merged["complexity"].astype(float), 1.0))
        tag = str(profile_tag) if profile_tag else "default"
        for col, label, suffix, color in self._leak_metric_specs():
            sub = merged[merged[col].notna()]
            if sub.empty:
                continue
            x = sub["topsim_intensional_levenshtein"].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            z = np.log2(np.maximum(sub["complexity"].to_numpy(dtype=float), 1.0))
            fig = plt.figure(figsize=(8, 6))
            sc = plt.scatter(x, y, c=z, cmap="viridis", alpha=0.85, edgecolors="none")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2 and np.ptp(x[mask]) > 1e-12:
                a, b = np.polyfit(x[mask], y[mask], deg=1)
                xg = np.linspace(x[mask].min(), x[mask].max(), 100)
                plt.plot(xg, a * xg + b, color="#111111", linewidth=2.0)
            r, p = self._pearson_r_p(x[mask], y[mask])
            r_txt = f"r={r:.3f}" if np.isfinite(r) else "r=nan"
            plt.title(f"topsim vs {label} ({tag}, {r_txt}, {self._format_p_value(p)})")
            plt.xlabel("topsim intensional levenshtein")
            plt.ylabel(label)
            plt.grid(True, alpha=0.3)
            cbar = plt.colorbar(sc)
            cbar.set_label("log2(complexity)")
            fig.tight_layout()
            fig.savefig(out_dir / f"topsim_vs_negation_{suffix}_{tag}_{self.name}.png", dpi=150)
            plt.close(fig)

    def plot_topsim_vs_n_f1(self, neg_df, *, control_col=None, profile_tag="", out_dir="plots"):
        df = self._prepare_negation_top_df(neg_df)
        topsim_df = self._build_topsim_df()
        if df is None or df.empty or topsim_df is None:
            return
        merged = df.merge(topsim_df, on="run_name", how="inner").dropna(subset=["topsim_intensional_levenshtein"])
        if merged.empty:
            return

        merged = merged.copy()
        if "F1_nT" not in merged.columns:
            n = pd.to_numeric(merged.get("n"), errors="coerce")
            l_t = pd.to_numeric(merged.get("l_T"), errors="coerce")
            den = n + l_t
            merged["F1_nT"] = np.where((den > 0) & n.notna() & l_t.notna(), 2.0 * n * l_t / den, np.nan)

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = str(profile_tag) if profile_tag else "default"
        metrics = [("n", "n"), ("F1_nT", "F1_nT")]
        x = pd.to_numeric(merged["topsim_intensional_levenshtein"], errors="coerce").to_numpy(dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for ax, (col, label) in zip(axes, metrics):
            y = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label)
                continue

            if control_col is not None and control_col in merged.columns and merged[control_col].notna().any():
                c = pd.to_numeric(merged[control_col], errors="coerce").to_numpy(dtype=float)
                c_mask = mask & np.isfinite(c)
                if c_mask.sum() >= 2:
                    sc = ax.scatter(x[c_mask], y[c_mask], c=c[c_mask], cmap="viridis", alpha=0.85, edgecolors="none")
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.set_label(control_col)
                else:
                    ax.scatter(x[mask], y[mask], alpha=0.85, edgecolors="none")
            else:
                ax.scatter(x[mask], y[mask], alpha=0.85, edgecolors="none")

            if np.ptp(x[mask]) > 1e-12:
                a, b = np.polyfit(x[mask], y[mask], deg=1)
                xg = np.linspace(x[mask].min(), x[mask].max(), 100)
                ax.plot(xg, a * xg + b, color="#111111", linewidth=2.0)

            corr, p = self._pearson_r_p(x[mask], y[mask])
            corr_txt = f"r={corr:.3f}" if np.isfinite(corr) else "r=nan"
            ax.set_title(f"{label} ({corr_txt}, {self._format_p_value(p)})")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("topsim intensional levenshtein")
            ax.set_ylabel(label)

        suptitle = "topsim correlation with n and F1_nT"
        if control_col:
            suptitle += f" by {control_col}"
        fig.suptitle(f"{suptitle} ({tag})")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        suffix = f"_by_{control_col}" if control_col else ""
        fig.savefig(out_dir / f"topsim_vs_n_f1{suffix}_{tag}_{self.name}.png", dpi=150)
        plt.close(fig)

    # ---- voc_reaper factorial plots (m2_resetandvoc and similar 2-factor experiments) ----

    def _build_final_metrics_df(self):
        """Per-run summary: final-epoch accuracy, topsim, reaper, voc, complexity."""
        topsim_priority = [
            "eval/topsim_extensional_norm_levenshtein",
            "eval/topsim_intensional_norm_levenshtein",
            "eval/topsim_extensional_levenshtein",
            "eval/topsim_intensional_levenshtein",
        ]
        rows = []
        for dp in self.datapoints:
            cfg = dp.get("config", {})
            eval_df = dp.get("evaluation")
            if eval_df is None or eval_df.empty:
                continue
            reaper = cfg.get("beth_reaper_step") or cfg.get("reaper_step")
            voc = cfg.get("voc_penalty", cfg.get("voc_pen"))
            complexity = cfg.get("num_predicates")
            last = eval_df.iloc[-1]
            acc = pd.to_numeric(last.get("eval/accuracy"), errors="coerce")
            topsim = np.nan
            for tc in topsim_priority:
                if tc in eval_df.columns:
                    v = pd.to_numeric(last.get(tc), errors="coerce")
                    if pd.notna(v):
                        topsim = v
                        break
            rows.append({
                "run_name": str(dp.get("run_name", "")),
                "complexity": complexity,
                "properties": str(cfg.get("properties")),
                "reaper_interval": reaper,
                "voc_penalty": voc,
                "final_accuracy": float(acc) if pd.notna(acc) else np.nan,
                "final_topsim": float(topsim) if pd.notna(topsim) else np.nan,
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        for c in ["complexity", "reaper_interval", "voc_penalty", "final_accuracy", "final_topsim"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _fmt_voc_label(self, v):
        if not np.isfinite(float(v)) or abs(float(v)) < 1e-15:
            return "0"
        return f"{float(v):.1e}"

    def _fmt_reaper_label(self, v):
        iv = int(float(v))
        return f"{iv}\n(no reset)" if iv >= 120 else str(iv)

    def _draw_heatmap_panel(self, ax, pivot, reaper_vals, voc_vals, *,
                            vmin, vmax, cmap="viridis",
                            baseline_reaper=120, baseline_voc=0.0):
        heat = pivot.reindex(index=reaper_vals, columns=voc_vals).values.astype(float)
        im = ax.imshow(heat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_xticks(range(len(voc_vals)))
        ax.set_xticklabels([self._fmt_voc_label(v) for v in voc_vals],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(reaper_vals)))
        ax.set_yticklabels([self._fmt_reaper_label(r) for r in reaper_vals], fontsize=7)
        midpoint = vmin + (vmax - vmin) * 0.5
        for ri in range(len(reaper_vals)):
            for vi in range(len(voc_vals)):
                val = heat[ri, vi]
                if np.isfinite(val):
                    txt_color = "white" if val < midpoint else "black"
                    ax.text(vi, ri, f"{val:.2f}", ha="center", va="center",
                            fontsize=5.5, color=txt_color)
        r_matches = [i for i, r in enumerate(reaper_vals) if int(float(r)) >= baseline_reaper]
        v_matches = [i for i, v in enumerate(voc_vals) if abs(float(v) - baseline_voc) < 1e-12]
        if r_matches and v_matches:
            ax.add_patch(plt.Rectangle((v_matches[0] - 0.5, r_matches[0] - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=2.0, zorder=5))
        return im

    def plot_pressure_heatmaps(self, *, out_dir="plots"):
        """Plots 1 & 2: 2D heatmaps of final accuracy and topsim over (reaper × voc), one panel per p."""
        df = self._build_final_metrics_df()
        if df is None or df.empty:
            print("[INFO] Pressure heatmaps skipped (no data).")
            return
        required = {"reaper_interval", "voc_penalty", "complexity"}
        if not required.issubset(df.columns):
            print("[INFO] Pressure heatmaps skipped (missing columns).")
            return
        sub = df.dropna(subset=["reaper_interval", "voc_penalty", "complexity"])
        if sub.empty:
            print("[INFO] Pressure heatmaps skipped (all NaN in grouping columns).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        complexities = sorted(sub["complexity"].unique())
        reaper_vals = sorted(sub["reaper_interval"].unique())
        voc_vals = sorted(sub["voc_penalty"].unique())

        metric_specs = []
        if "final_accuracy" in sub.columns and sub["final_accuracy"].notna().any():
            metric_specs.append(("final_accuracy", "accuracy", "accuracy", "Blues"))
        if "final_topsim" in sub.columns and sub["final_topsim"].notna().any():
            metric_specs.append(("final_topsim", "topographic similarity", "topsim", "viridis"))

        for metric_col, metric_label, file_suffix, cmap in metric_specs:
            valid = sub.dropna(subset=[metric_col])
            if valid.empty:
                continue
            vmin = float(np.nanpercentile(valid[metric_col], 5))
            vmax = float(np.nanpercentile(valid[metric_col], 95))
            ncols = len(complexities)
            row_h = max(5, 0.45 * len(reaper_vals) + 1.5)
            fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, row_h), squeeze=False)
            for j, c in enumerate(complexities):
                ax = axes[0][j]
                csub = valid[valid["complexity"] == c]
                heat_agg = (csub.groupby(["reaper_interval", "voc_penalty"])[metric_col]
                            .median().reset_index())
                pivot = heat_agg.pivot(index="reaper_interval", columns="voc_penalty",
                                       values=metric_col)
                im = self._draw_heatmap_panel(ax, pivot, reaper_vals, voc_vals,
                                              vmin=vmin, vmax=vmax, cmap=cmap)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                props = csub["properties"].dropna().astype(str).unique()
                props_str = (sorted(props, key=self._properties_sort_key)[0]
                             if len(props) else str(int(c)))
                ax.set_title(f"p = {props_str}", fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel("reset interval  (epochs between resets)", fontsize=8)
                ax.set_xlabel("voc_penalty", fontsize=8)
            fig.suptitle(
                f"Median {metric_label} by reset interval × vocabulary pressure\n"
                "(red outline = no-pressure baseline)",
                fontsize=11, fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"pressure_heatmap_{file_suffix}_{self.name}.png", dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved pressure_heatmap_{file_suffix}_{self.name}.png")

    def plot_topsim_interaction_reaper_voc(self, *, out_dir="plots"):
        """Plot 3: topsim vs reaper_step, one line per voc_penalty, panels for p."""
        df = self._build_final_metrics_df()
        if df is None or df.empty:
            return
        if "final_topsim" not in df.columns or df["final_topsim"].isna().all():
            print("[INFO] Interaction plot skipped (no topsim data).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sub = df.dropna(subset=["reaper_interval", "voc_penalty", "complexity", "final_topsim"])
        if sub.empty:
            print("[INFO] Interaction plot skipped (no valid rows).")
            return

        complexities = sorted(sub["complexity"].unique())
        voc_vals = sorted(sub["voc_penalty"].unique())
        palette = sns.color_palette("plasma", n_colors=max(1, len(voc_vals)))
        voc_colors = {v: palette[i] for i, v in enumerate(voc_vals)}

        ncols = len(complexities)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5), squeeze=False, sharey=True)

        for j, c in enumerate(complexities):
            ax = axes[0][j]
            csub = sub[sub["complexity"] == c]
            if csub.empty:
                continue
            active_reapers = sorted([r for r in csub["reaper_interval"].unique() if r < 120])

            for v in voc_vals:
                color = voc_colors[v]
                label = self._fmt_voc_label(v)
                v_arr = csub["voc_penalty"].to_numpy(dtype=float)
                vsub = csub[np.isclose(v_arr, float(v), atol=1e-12)]
                active = vsub[vsub["reaper_interval"] < 120]
                if not active.empty:
                    stats = (
                        active.groupby("reaper_interval")["final_topsim"]
                        .agg(median="median",
                             q25=lambda x: np.nanquantile(x, 0.25),
                             q75=lambda x: np.nanquantile(x, 0.75))
                        .reset_index().sort_values("reaper_interval")
                    )
                    ax.fill_between(stats["reaper_interval"], stats["q25"], stats["q75"],
                                    alpha=0.12, color=color)
                    ax.plot(stats["reaper_interval"], stats["median"],
                            marker="o", linewidth=2.0, color=color, markersize=5, label=label)
                no_reset = vsub[vsub["reaper_interval"] >= 120]
                if not no_reset.empty:
                    baseline_val = float(no_reset["final_topsim"].median())
                    ax.axhline(baseline_val, linestyle="--", color=color,
                               linewidth=1.2, alpha=0.6)

            if active_reapers:
                ax.set_xscale("log", base=2)
                ax.set_xticks(active_reapers)
                ax.set_xticklabels([str(int(r)) for r in active_reapers],
                                   rotation=45, ha="right", fontsize=8)
            props = csub["properties"].dropna().astype(str).unique()
            props_str = (sorted(props, key=self._properties_sort_key)[0]
                         if len(props) else str(int(c)))
            ax.set_title(f"p = {props_str}", fontsize=10, fontweight="bold")
            ax.set_xlabel("reset interval (epochs)", fontsize=9)
            if j == 0:
                ax.set_ylabel("final topsim  (median ± IQR)", fontsize=9)
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0][0].get_legend_handles_labels()
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], linestyle="--", color="gray", linewidth=1.2))
        labels.append("no-reset baseline")
        fig.legend(handles, labels, title="voc_penalty", loc="upper center",
                   bbox_to_anchor=(0.5, 1.01), ncol=min(8, len(handles)), fontsize=8)
        fig.suptitle(
            "Topsim vs. reset interval  (dashed = no-reset per voc_penalty)",
            fontsize=11, fontweight="bold", y=1.07,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"topsim_interaction_reaper_voc_{self.name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved topsim_interaction_reaper_voc_{self.name}.png")

    def plot_pressure_heatmaps_negation(self, neg_df, *, out_dir="plots"):
        """Plot 4: F1_nT (strong) and F1_nr (weak) heatmaps over (reaper × voc), one col per p."""
        df_neg = self._prepare_negation_top_df(neg_df)
        if df_neg is None or df_neg.empty:
            print("[INFO] Negation heatmaps skipped (no negation data).")
            return
        required = {"reaper_interval", "voc_penalty", "complexity"}
        if not required.issubset(df_neg.columns):
            print("[INFO] Negation heatmaps skipped (missing grouping columns).")
            return
        df_neg = df_neg.dropna(subset=["reaper_interval", "voc_penalty", "complexity"]).copy()
        if df_neg.empty:
            return

        if "F1_nT" not in df_neg.columns:
            n = pd.to_numeric(df_neg.get("n"), errors="coerce")
            t = pd.to_numeric(df_neg.get("l_T", df_neg.get("t")), errors="coerce")
            non_t = 1.0 - t
            den = n + non_t
            df_neg["F1_nT"] = np.where(
                (den > 0) & n.notna() & t.notna(), 2.0 * n * non_t / den, np.nan)
        if "F1_nr" not in df_neg.columns:
            n = pd.to_numeric(df_neg.get("n"), errors="coerce")
            r = pd.to_numeric(df_neg.get("r", df_neg.get("c")), errors="coerce")
            den = n + r
            df_neg["F1_nr"] = np.where(
                (den > 0) & n.notna() & r.notna(), 2.0 * n * r / den, np.nan)

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        complexities = sorted(pd.to_numeric(df_neg["complexity"], errors="coerce").dropna().unique())
        reaper_vals = sorted(pd.to_numeric(df_neg["reaper_interval"], errors="coerce").dropna().unique())
        voc_vals = sorted(pd.to_numeric(df_neg["voc_penalty"], errors="coerce").dropna().unique())

        f_specs = [
            ("F1_nT", r"$F_{1,nT}$  (strong negation: n $\times$ non-entanglement)"),
            ("F1_nr", r"$F_{1,nr}$  (weak negation: n $\times$ conservation)"),
        ]
        available_f = [(col, lbl) for col, lbl in f_specs
                       if col in df_neg.columns and df_neg[col].notna().any()]
        if not available_f:
            print("[INFO] Negation heatmaps skipped (no F1_nT or F1_nr data).")
            return

        nrows = len(available_f)
        ncols = len(complexities)
        row_h = max(4, 0.42 * len(reaper_vals) + 1.0)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, row_h * nrows),
                                  squeeze=False)

        all_vals = pd.concat([df_neg[col] for col, _ in available_f if col in df_neg.columns],
                             ignore_index=True).dropna()
        vmin = float(np.nanpercentile(all_vals, 5)) if not all_vals.empty else 0.0
        vmax = float(np.nanpercentile(all_vals, 95)) if not all_vals.empty else 1.0

        for i, (col, f_label) in enumerate(available_f):
            for j, c in enumerate(complexities):
                ax = axes[i][j]
                csub = df_neg[df_neg["complexity"] == c].dropna(subset=[col])
                if csub.empty:
                    ax.set_visible(False)
                    continue
                heat_agg = (csub.groupby(["reaper_interval", "voc_penalty"])[col]
                            .median().reset_index())
                pivot = heat_agg.pivot(index="reaper_interval", columns="voc_penalty", values=col)
                im = self._draw_heatmap_panel(ax, pivot, reaper_vals, voc_vals,
                                              vmin=vmin, vmax=vmax, cmap="magma")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if i == 0:
                    props = (csub["properties"].dropna().astype(str).unique()
                             if "properties" in csub.columns else [])
                    props_str = (sorted(props, key=self._properties_sort_key)[0]
                                 if len(props) else str(int(c)))
                    ax.set_title(f"p = {props_str}", fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(f"{f_label}\nreset interval", fontsize=7)
                if i == nrows - 1:
                    ax.set_xlabel("voc_penalty", fontsize=8)

        fig.suptitle(
            "Negation compositionality by reset interval × vocabulary pressure\n"
            "(red outline = no-pressure baseline)",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"pressure_heatmap_negation_{self.name}.png", dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved pressure_heatmap_negation_{self.name}.png")

    def _message_agreement(self, lang_a, lang_b):
        """Fraction of predicates with identical messages between two language dumps."""
        for df in (lang_a, lang_b):
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return np.nan
            if "pred_str" not in df.columns or "msg" not in df.columns:
                return np.nan

        def _modal(df):
            return (df.groupby("pred_str")["msg"]
                    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
                    .reset_index())

        def _norm(s):
            return " ".join(t for t in str(s).split() if t != "0")

        a_agg = _modal(lang_a)
        b_agg = _modal(lang_b)
        merged = a_agg.merge(b_agg, on="pred_str", suffixes=("_a", "_b"))
        if merged.empty:
            return np.nan
        merged["msg_a"] = merged["msg_a"].apply(_norm)
        merged["msg_b"] = merged["msg_b"].apply(_norm)
        return float((merged["msg_a"] == merged["msg_b"]).mean())

    def plot_intergenerational_stability(self, *, out_dir="plots"):
        """Plot 5: language agreement across generation boundaries vs. generation index.

        Tests the iterated-learning prediction: if reaper pressure shapes the code to
        be more relearnable, agreement between consecutive-generation language dumps
        should increase over training.
        """
        rows = []
        for dp in self.datapoints:
            cfg = dp.get("config", {})
            langs = dp.get("languages") or []
            reaper = cfg.get("beth_reaper_step") or cfg.get("reaper_step")
            voc = cfg.get("voc_penalty", cfg.get("voc_pen"))
            complexity = cfg.get("num_predicates")
            if reaper is None or float(reaper) >= 120:
                continue
            if len(langs) < 2:
                continue
            K = int(float(reaper))
            sorted_langs = sorted(langs, key=lambda x: int(x.get("epoch_number", 0)))
            for ld in sorted_langs:
                ld["_gen"] = int(ld.get("epoch_number", 0)) // K
            for idx in range(len(sorted_langs) - 1):
                d_curr = sorted_langs[idx]
                d_next = sorted_langs[idx + 1]
                if d_next["_gen"] <= d_curr["_gen"]:
                    continue
                agr = self._message_agreement(d_curr["language"], d_next["language"])
                if np.isfinite(agr):
                    rows.append({
                        "run_name": str(dp.get("run_name", "")),
                        "complexity": complexity,
                        "properties": str(cfg.get("properties")),
                        "reaper_interval": reaper,
                        "voc_penalty": voc,
                        "gen_from": d_curr["_gen"],
                        "agreement": agr,
                    })

        if not rows:
            print("[INFO] Intergenerational stability plot skipped (no cross-generation language pairs).")
            return

        df = pd.DataFrame(rows)
        for c in ["complexity", "reaper_interval", "voc_penalty", "gen_from", "agreement"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["complexity", "reaper_interval", "voc_penalty",
                                "gen_from", "agreement"])
        if df.empty:
            print("[INFO] Intergenerational stability plot skipped (no valid rows after coercion).")
            return

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"intergenerational_stability_{self.name}.csv", index=False)

        complexities = sorted(df["complexity"].unique())
        voc_vals = sorted(df["voc_penalty"].unique())
        reaper_vals = sorted(df["reaper_interval"].unique())
        palette = sns.color_palette("plasma", n_colors=max(1, len(voc_vals)))
        voc_colors = {v: palette[i] for i, v in enumerate(voc_vals)}

        n_reap = len(reaper_vals)
        n_comp = len(complexities)
        fig, axes = plt.subplots(n_reap, n_comp, figsize=(4.5 * n_comp, 3.5 * n_reap),
                                  squeeze=False, sharex="col", sharey=True)
        for ri, r in enumerate(reaper_vals):
            for ci, c in enumerate(complexities):
                ax = axes[ri][ci]
                sub = df[(df["reaper_interval"] == r) & (df["complexity"] == c)]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                for v in voc_vals:
                    v_arr = sub["voc_penalty"].to_numpy(dtype=float)
                    vsub = sub[np.isclose(v_arr, float(v), atol=1e-12)]
                    if vsub.empty:
                        continue
                    stats = (
                        vsub.groupby("gen_from")["agreement"]
                        .agg(median="median",
                             q25=lambda x: np.nanquantile(x, 0.25),
                             q75=lambda x: np.nanquantile(x, 0.75))
                        .reset_index().sort_values("gen_from")
                    )
                    color = voc_colors[v]
                    ax.fill_between(stats["gen_from"], stats["q25"], stats["q75"],
                                    alpha=0.12, color=color)
                    ax.plot(stats["gen_from"], stats["median"], marker="o", linewidth=1.8,
                            color=color, markersize=4, label=self._fmt_voc_label(v))
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                if ri == 0:
                    props = (sub["properties"].dropna().astype(str).unique()
                             if "properties" in sub.columns else [])
                    props_str = (sorted(props, key=self._properties_sort_key)[0]
                                 if len(props) else str(int(c)))
                    ax.set_title(f"p = {props_str}", fontsize=9, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(f"reset={int(r)}\nagreement", fontsize=8)
                if ri == n_reap - 1:
                    ax.set_xlabel("generation index", fontsize=8)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="voc_penalty", loc="upper right", fontsize=7)
        fig.suptitle(
            "Language agreement across generation boundaries\n"
            "(iterated learning predicts: agreement increases over generations)",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"intergenerational_stability_{self.name}.png", dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved intergenerational_stability_{self.name}.png")

    def compute_optimal_table(self, neg_df=None, *, accuracy_threshold=0.9, out_dir="outputs"):
        """Summary table: optimal (reaper, voc) per p maximising topsim s.t. accuracy >= threshold × baseline."""
        df = self._build_final_metrics_df()
        if df is None or df.empty:
            print("[INFO] Optimal table skipped (no data).")
            return None
        required = {"reaper_interval", "voc_penalty", "complexity", "final_accuracy", "final_topsim"}
        if not required.issubset(df.columns):
            print("[INFO] Optimal table skipped (missing columns).")
            return None

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Optional negation F1 summary per (reaper, voc, complexity)
        neg_summary = None
        if neg_df is not None:
            df_neg = self._prepare_negation_top_df(neg_df)
            if df_neg is not None and not df_neg.empty:
                df_neg = df_neg.copy()
                if "F1_nT" not in df_neg.columns:
                    n = pd.to_numeric(df_neg.get("n"), errors="coerce")
                    t = pd.to_numeric(df_neg.get("l_T", df_neg.get("t")), errors="coerce")
                    non_t = 1.0 - t
                    den = n + non_t
                    df_neg["F1_nT"] = np.where(
                        (den > 0) & n.notna() & t.notna(), 2.0 * n * non_t / den, np.nan)
                if "F1_nr" not in df_neg.columns:
                    n = pd.to_numeric(df_neg.get("n"), errors="coerce")
                    r = pd.to_numeric(df_neg.get("r", df_neg.get("c")), errors="coerce")
                    den = n + r
                    df_neg["F1_nr"] = np.where(
                        (den > 0) & n.notna() & r.notna(), 2.0 * n * r / den, np.nan)
                neg_cols = [c for c in ["F1_nT", "F1_nr"]
                            if c in df_neg.columns and df_neg[c].notna().any()]
                if neg_cols and {"reaper_interval", "voc_penalty", "complexity"}.issubset(df_neg.columns):
                    neg_summary = (df_neg.groupby(["reaper_interval", "voc_penalty", "complexity"])
                                   [neg_cols].median().reset_index())

        complexities = sorted(df["complexity"].dropna().unique())
        table_rows = []
        for c in complexities:
            csub = df[df["complexity"] == c].dropna(
                subset=["reaper_interval", "voc_penalty", "final_accuracy", "final_topsim"])
            if csub.empty:
                continue
            cell_agg = (csub.groupby(["reaper_interval", "voc_penalty"])
                        .agg(acc_med=("final_accuracy", "median"),
                             topsim_med=("final_topsim", "median"))
                        .reset_index())
            baseline_rows = cell_agg[(cell_agg["reaper_interval"] >= 120) &
                                      (cell_agg["voc_penalty"].abs() < 1e-12)]
            baseline_acc = (float(baseline_rows["acc_med"].iloc[0])
                            if not baseline_rows.empty else float(cell_agg["acc_med"].max()))
            eligible = cell_agg[cell_agg["acc_med"] >= accuracy_threshold * baseline_acc]
            if eligible.empty:
                eligible = cell_agg
            best = eligible.loc[eligible["topsim_med"].idxmax()]
            props_vals = csub["properties"].dropna().unique() if "properties" in csub.columns else []
            p_label = props_vals[0] if len(props_vals) > 0 else str(int(c))
            row = {
                "p": p_label,
                "reaper_step": int(float(best["reaper_interval"])),
                "voc_penalty": float(best["voc_penalty"]),
                "accuracy": round(float(best["acc_med"]), 3),
                "topsim": round(float(best["topsim_med"]), 3),
                "baseline_accuracy": round(baseline_acc, 3),
            }
            if neg_summary is not None:
                match = neg_summary[
                    (neg_summary["complexity"] == c) &
                    (neg_summary["reaper_interval"] == best["reaper_interval"]) &
                    (neg_summary["voc_penalty"] - best["voc_penalty"]).abs() < 1e-12
                ]
                for col in ["F1_nT", "F1_nr"]:
                    row[col] = (round(float(match[col].iloc[0]), 3)
                                if not match.empty and col in match.columns else np.nan)
            table_rows.append(row)

        if not table_rows:
            print("[INFO] Optimal table: no valid rows.")
            return None

        tbl = pd.DataFrame(table_rows)
        out_path = out_dir / f"optimal_pressure_config_{self.name}.csv"
        tbl.to_csv(out_path, index=False)
        print(f"[INFO] Optimal pressure configuration:")
        print(tbl.to_string(index=False))
        print(f"[INFO] Saved → {out_path}")
        return tbl

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
        plt.close(fig)
