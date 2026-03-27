import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

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
            sns.lineplot(data=plot_df, x='complexity', y='epochs_to_max_acc',
                         marker='o', err_style='band', label="max accuracy")
            if plot_df["epochs_to_max_perf"].notna().any():
                sns.lineplot(data=plot_df, x='complexity', y='epochs_to_max_perf',
                             marker='o', err_style='band', label="max performance")
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
            sns.lineplot(
                data=long_df,
                x="complexity",
                y="epochs_to_max",
                hue=hue,
                style="metric",
                markers=True,
                dashes=False,
                err_style="band",
            )

        plt.title('epochs until convergence by number of predicates', fontweight='bold')
        plt.xlabel('number of predicates')
        plt.ylabel('epochs to max')
        plt.grid(True, alpha=0.3)
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
