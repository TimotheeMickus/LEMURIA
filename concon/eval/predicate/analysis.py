import os, pathlib, sys
import argparse
import pickle
import pandas as pd
import load
import negation
import plots

if __name__ == "__main__":
    # Ask for experimental data
    # Assumes that all runs for one experiment are written into a dedicated folder
    # And that that folder is inside "runs/"
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    runs_dir = repo_root / "runs"
    print(f"Available experiments: {os.listdir(runs_dir)}")
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", help="experiment name under runs/")
    parser.add_argument("--negation", action="store_true", help="run negation analysis")
    parser.add_argument("--negation-top-rows", type=int, default=10, help="number of top rows per run")
    parser.add_argument("--feat-operator", choices=["or", "and"], default="or", help="feature composition operator for negation analysis")
    parser.add_argument("--latest-only", action="store_true", help="analyze only latest language per run (default: analyze all)")
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel jobs for negation export")
    parser.add_argument("--plots", action="store_true", help="generate plots")
    args = parser.parse_args()
    experiment_path = args.experiment or input("Experiment name: ")

    # ----- DEBUG / TESTING ONLY -----

    has_eval, has_pred, has_lang = 0, 0, 0
    total = 0
    vocab_sum = 0
    vocab_count = 0
    cache_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"datapoints_{experiment_path}.pkl"

    if cache_path.is_file():
        print(f"Loading cached datapoints from {cache_path} ...")
        try:
            with open(cache_path, "rb") as f:
                datapoints_list = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load cache ({e}). Re-loading datapoints.")
            datapoints_list = load.get_datapoints(experiment_path)
    else:
        print("Loading datapoints...")
        datapoints_list = load.get_datapoints(experiment_path)
        with open(cache_path, "wb") as f:
            pickle.dump(datapoints_list, f)
        print(f"Saved datapoint cache to {cache_path}")
    for d in datapoints_list:
        total += 1
        if d['evaluation'] is not None:
            has_eval += 1
            if "eval/vocab_used" in d["evaluation"].columns and not d["evaluation"].empty:
                vocab_sum += d['evaluation']['eval/vocab_used'].iloc[-1]
                vocab_count += 1
        if d['predicates'] is not None:
            has_pred += 1
        if d['languages']:
            has_lang += 1
    print(f"Found {total} datapoints.")
    print(f"Of which {has_eval} have eval, {has_pred} have pred, {has_lang} have lang.")
    print(f"Average vocab length: {vocab_sum/vocab_count if vocab_count else 'n/a'}")

    neg_top1_df = None
    if args.negation:
        _, neg_top1_df = negation.export_analysis(
            datapoints_list,
            experiment_name=experiment_path,
            top_rows=args.negation_top_rows,
            operator=args.feat_operator,
            latest_only=args.latest_only,
            n_jobs=args.n_jobs,
            outputs_dir=pathlib.Path(__file__).resolve().parent / "outputs",
        )

    cm = plots.ComplexityMemory(datapoints_list, name=experiment_path)
    if args.plots:
        out_dir = pathlib.Path(__file__).resolve().parent / "plots"
        # Plot: epochs to max accuracy/performance
        cm.plot_time_to_max_accuracy(out_dir=out_dir)
        cm.plot_message_compression(group_by=("properties",), out_dir=out_dir)
        cm.plot_message_compression(group_by=("hidden_size",), out_dir=out_dir)

        if neg_top1_df is None:
            top1_csv = cache_dir / f"negation_top_1_{experiment_path}.csv"
            if top1_csv.is_file():
                neg_top1_df = pd.read_csv(top1_csv)
                print(f"Loaded existing negation top-1 for plots: {top1_csv}")

        if neg_top1_df is not None and not neg_top1_df.empty:
            cm.plot_negation_metrics_by_complexity(neg_top1_df, out_dir=out_dir)
            cm.plot_negation_metric_slopes(neg_top1_df, profile_tag=args.feat_operator, out_dir=out_dir)
            cm.plot_topsim_vs_negation(neg_top1_df, profile_tag=args.feat_operator, out_dir=out_dir)
            cm.plot_topsim_interaction_n(neg_top1_df, profile_tag=args.feat_operator, out_dir=out_dir)
        else:
            print(f"[INFO] No negation top-1 dataframe available for negation-specific plots.")
    # cm.plot_message_compression(group_by=("negation","properties"))
