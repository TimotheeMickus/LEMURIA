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
    parser.add_argument("--cache", action="store_true", help="use/load datapoint cache")
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel workers for negation analysis (capped at 4)")
    parser.add_argument("--plots", action="store_true", help="generate plots")
    args = parser.parse_args()
    experiment_path = args.experiment or input("Experiment name: ")

    # ----- DEBUG / TESTING ONLY -----

    has_eval, has_pred, has_lang = 0, 0, 0
    total = 0
    vocab_sum = 0
    vocab_count = 0
    latest_language_only = True
    cache_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_latestlang" if latest_language_only else ""
    cache_path = cache_dir / f"datapoints_{experiment_path}{cache_suffix}.pkl"

    if args.cache and cache_path.is_file():
        print(f"Loading cached datapoints from {cache_path} ...")
        try:
            with open(cache_path, "rb") as f:
                datapoints_list = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load cache ({e}). Re-loading datapoints and rebuilding cache.")
            datapoints_list = load.get_datapoints(experiment_path, latest_language_only=latest_language_only)
            with open(cache_path, "wb") as f:
                pickle.dump(datapoints_list, f)
            print(f"Saved datapoint cache to {cache_path}")
    elif args.cache:
        datapoints_list = load.get_datapoints(experiment_path, latest_language_only=latest_language_only)
        with open(cache_path, "wb") as f:
            pickle.dump(datapoints_list, f)
        print(f"Saved datapoint cache to {cache_path}")
    else:
        datapoints_list = load.get_datapoints(experiment_path, latest_language_only=latest_language_only)
    for d in datapoints_list:
        total += 1
        avg_used_vocab = 0
        if d['evaluation'] is not None:
            has_eval += 1
            vocab_sum += d['evaluation']['eval/vocab_used'].iloc[-1]
            vocab_count += 1
        if d['predicates'] is not None:
            has_pred += 1
        if d['languages']:
            has_lang += 1
    print(f"Found {total} datapoints.")
    print(f"Of which {has_eval} have eval, {has_pred} have pred, {has_lang} have lang.")
    avg_vocab = (vocab_sum / vocab_count) if vocab_count else float("nan")
    print(f"Average vocab length: {avg_vocab}")

    analysis_df = None
    summary_df = None
    if args.negation:
        negation_datapoints = [
            d for d in datapoints_list
            if not d.get("config", {}).get("no_negation", False)
        ]
        print(f"Negation-enabled runs: {len(negation_datapoints)}/{len(datapoints_list)}")
        n_jobs = min(4, max(1, int(args.n_jobs)))
        print(f"Negation n_jobs: {n_jobs} (cap=4)")
        print("Running negation export (single-symbol mode)")
        analysis_df, summary_df, analysis_path, summary_path = negation.export_negation_metrics_csvs(
            negation_datapoints,
            top_k=5,
            out_dir=cache_dir,
            experiment_name=experiment_path,
            expansion_rounds=0,
            n_jobs=n_jobs,
            bin_profile="single_symbol",
        )
        print(f"Saved negation top-k analysis to: {analysis_path}")
        print(f"Saved negation top-1 summary to: {summary_path}")
        print(f"Rows: analysis={len(analysis_df)}, summary={len(summary_df)}")

        if args.plots:
            out_dir = pathlib.Path(__file__).resolve().parent / "plots"
            cm_neg = plots.ComplexityMemory(datapoints_list, name=experiment_path)
            cm_neg.plot_negation_metrics_by_complexity(summary_df, out_dir=out_dir)
            cm_neg.plot_negation_metric_slopes(summary_df, profile_tag="single_symbol", out_dir=out_dir)
            cm_neg.plot_topsim_vs_negation(summary_df, profile_tag="single_symbol", out_dir=out_dir)
            cm_neg.plot_topsim_interaction_xor(summary_df, profile_tag="single_symbol", out_dir=out_dir)

    if args.plots:
        cm = plots.ComplexityMemory(datapoints_list, name=experiment_path)
        out_dir = pathlib.Path(__file__).resolve().parent / "plots"
        # Always run the non-negation plot suite once for this experiment.
        cm.plot_requested_suite(neg_df=None, out_dir=out_dir)

        if not args.negation:
            summary_csv = cache_dir / f"negation_summary_top1_{experiment_path}.csv"
            if summary_csv.is_file():
                try:
                    cached_df = pd.read_csv(summary_csv)
                    print(f"Loaded existing negation summary for plots: {summary_csv}")
                    cm_neg = plots.ComplexityMemory(datapoints_list, name=experiment_path)
                    cm_neg.plot_negation_metrics_by_complexity(cached_df, out_dir=out_dir)
                    cm_neg.plot_negation_metric_slopes(cached_df, profile_tag="single_symbol", out_dir=out_dir)
                    cm_neg.plot_topsim_vs_negation(cached_df, profile_tag="single_symbol", out_dir=out_dir)
                    cm_neg.plot_topsim_interaction_xor(cached_df, profile_tag="single_symbol", out_dir=out_dir)
                except Exception as e:
                    print(f"[WARN] Failed to load cached negation summary ({e}).")
            else:
                print(f"[INFO] No cached negation summary found at {summary_csv}.")
