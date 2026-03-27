import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# ------ complexity - memory interactions ------

class complexityMemory:
    # Runs differ by hidden size (48, 96, 192).
    # All have depth 1 - 2.
    # There are varying "n" property kinds with negation and no_negation.
    # There are varying "n-n" property kinds with conjunction.
    # Each run is repeated 5 times.
    # All other hyperparameters are equal.
    def __init__(self, datapoints, name=None):
        self.datapoints = datapoints
        self.name = name or "unnamed"

    def plot_time_to_max_accuracy(self, out_dir="plots"):
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
                    'epochs_to_max_acc': best_epoch_acc,
                    'epochs_to_max_perf': best_epoch_perf,
                })

        if(missing): print(f"[WARN] Missing at least {missing} evaluation data.")

        if not results:
            print(f"No valid evaluation data found in {self.name}.")
        else:
            plot_df = pd.DataFrame(results)

            # Visualization
            fig = plt.figure(figsize=(10, 6))
            sns.lineplot(data=plot_df, x='total predicates', y='epoch to max acc.', marker='o', err_style='band', label="max accuracy")
            if plot_df["epochs_to_max_perf"].notna().any():
                sns.lineplot(data=plot_df, x='total predicates', y='epoch to max perf.', marker='o', err_style='band', label="max performance")

            plt.title('epochs until convergence by number of predicates', fontweight='bold')
            plt.xlabel('number of predicates')
            plt.ylabel('epochs to max')
            plt.grid(True, alpha=0.3)
            xticks = sorted(plot_df['complexity'].unique())
            xtick_labels = []
            for c in xticks:
                props = plot_df.loc[plot_df["complexity"] == c, "properties"].dropna().astype(int).unique()
                props_str = props[0] if len(props) else "?"
                xtick_labels.append(f"{c}: {props_str}")
            plt.xscale("log", base=2)
            plt.xticks(xticks, xtick_labels, rotation=45, ha="right")

            out_dir = pathlib.Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"epoch_to_max_metrics_{self.name}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)

            # Plot stratified by hidden size
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(1, 1, 1)
            sns.lineplot(data=plot_df, x='total predicates', y='epochs to max', hue='network capacity',
                         marker='o', err_style='band', ax=ax2)
            ax2.set_title('epochs until convergence by number of predicates', fontweight='bold')
            ax2.set_xlabel('number of predicates')
            ax2.set_ylabel('epochs to max')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale("log", base=2)
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xtick_labels, rotation=45, ha="right")
            out_path2 = out_dir / f"epoch_to_max_acc_by_hidden_{self.name}.png"
            fig2.tight_layout()
            fig2.savefig(out_path2, dpi=150)

            # Plot stratified by +-negation
            fig3 = plt.figure(figsize=(10, 6))
            ax3 = fig3.add_subplot(1, 1, 1)
            sns.lineplot(data=plot_df, x='number of predicates', y='epochs to max', hue='negation',
                         marker='o', err_style='band', ax=ax3)
            ax3.set_title('epochs until convergence by number of predicates', fontweight='bold')
            ax3.set_xlabel('number of predicates')
            ax3.set_ylabel('epochs to max')
            ax3.grid(True, alpha=0.3)
            ax3.set_xscale("log", base=2)
            ax3.set_xticks(xticks)
            ax3.set_xticklabels([str(c) for c in xticks], rotation=45, ha="right")
            out_path3 = out_dir / f"epoch_to_max_acc_by_negation_{self.name}.png"
            fig3.tight_layout()
            fig3.savefig(out_path3, dpi=150)

            summary = plot_df.groupby('complexity')[['epochs_to_max_acc', 'epochs_to_max_perf']].agg(['mean', 'std', 'count'])
            print(summary)
