import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    def plot_time_to_max_accuracy(datapoints: dict):
        # TODO
        '''
        datapoints (list[dict]) of runs with keys:
            - config (dict), 
            - evaluation (DataFrame), 
            - predicates (DataFrame),
            - languages (list[dict[epoch_number: int, DataFrame]])
        '''
        results = []
        missing = 0

        # Iterate through datapoints
        for dp in datapoints:
            eval_df = dp.get("evaluation")
            if eval_df is None or "eval/accuracy" not in eval_df.columns:
                missing += 1
                continue

            complexity = dp["config"].get("num_predicates")
            properties = dp["config"].get("properties")
            hidden     = dp["config"].get("hidden_size")
            # Find the maximum accuracy and performance
            max_acc    = eval_df["eval/accuracy"].max()
            max_perf   = eval_df["eval/perf"].max() if "eval/perf" in eval_df.columns else None
            # Get the first epoch where this max was achieved (within tolerance).
            best_epoch_acc = eval_df.loc[eval_df["eval/accuracy"] >= (max_acc - 0.001), "epoch"].min()
            best_epoch_perf = eval_df.loc[eval_df["eval/perf"] >= (max_perf - 0.001), "epoch"]
            results.append({
                    'Complexity': int(complexity),
                    'Properties': properties,
                    'Network capacity': hidden,
                    'Max Accuracy': max_acc,
                    'Max Performance': max_perf,
                    'Epoch to Max Acc.': best_epoch_acc,
                    'Epoch to Max Perf.': best_epoch_perf,
                })

        if missing:
            print(f"[WARN] Missing at least {missing} evaluation data.")

        if not results:
            print("No valid evaluation data found in sweep_data.")
        else:
            plot_df = pd.DataFrame(results)

            # Visualization
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=plot_df, x='Complexity', y='Epoch to Max', marker='o', err_style='band')
            sns.scatterplot(data=plot_df, x='Complexity', y='Epoch to Max', alpha=0.5)

            plt.title('Time to Maximum Accuracy vs. Game Complexity', fontweight='bold')
            plt.xlabel('Complexity (Number of Predicates)')
            plt.ylabel('Epoch to Reach Max Accuracy')
            plt.grid(True, alpha=0.3)
            plt.xticks(sorted(plot_df['Complexity'].unique()))
            plt.show()

            display(plot_df.groupby('Complexity')['Epoch to Max'].agg(['mean', 'std', 'count']))
