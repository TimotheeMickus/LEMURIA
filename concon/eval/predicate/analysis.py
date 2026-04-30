import os, pathlib, sys
import argparse
import pickle
import gc
import pandas as pd
import numpy as np
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
    parser.add_argument("--negation-profile", choices=["fast", "slow"], default="fast", help="search profile for negation feature growth")
    parser.add_argument("--feat-operator", choices=["or", "and"], default="or", help="feature composition operator for negation analysis")
    parser.add_argument("--latest-only", action="store_true", help="analyze only latest language per run (default: analyze all)")
    parser.add_argument("--best-accuracy", action="store_true", help="analyze only the language dump closest to the first epoch that reached max eval accuracy")
    parser.add_argument("--min-eval-accuracy", type=float, default=None, help="analyze only runs whose max eval/accuracy is >= threshold (e.g. 0.95)")
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel jobs for negation export")
    parser.add_argument("--plots", action="store_true", help="generate plots")
    args = parser.parse_args()
    if args.latest_only and args.best_accuracy:
        parser.error("--latest-only and --best-accuracy are mutually exclusive.")
    if args.min_eval_accuracy is not None and not (0.0 <= args.min_eval_accuracy <= 1.0):
        parser.error("--min-eval-accuracy must be in [0, 1].")
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

    def _max_eval_accuracy(dp):
        ev = dp.get("evaluation")
        if ev is None or ev.empty or "eval/accuracy" not in ev.columns:
            return np.nan
        vals = pd.to_numeric(ev["eval/accuracy"], errors="coerce")
        return float(vals.max()) if vals.notna().any() else np.nan

    def _best_accuracy_epoch(eval_df):
        if eval_df is None or eval_df.empty: return None
        if "eval/accuracy" not in eval_df.columns or "epoch" not in eval_df.columns: return None
        acc = pd.to_numeric(eval_df["eval/accuracy"], errors="coerce")
        ep = pd.to_numeric(eval_df["epoch"], errors="coerce")
        valid = acc.notna() & ep.notna()
        if not valid.any(): return None
        acc_v = acc[valid]
        ep_v = ep[valid]
        m = float(acc_v.max())
        hit = (acc_v - m).abs() <= 1e-12
        if not hit.any(): return None
        return int(ep_v.loc[hit].iloc[0])

    def _select_language_for_epoch(langs, target_epoch):
        if not langs: return []
        if target_epoch is None: return [max(langs, key=lambda x: x.get("epoch_number", -1))]
        exact = [x for x in langs if x.get("epoch_number", None) == target_epoch]
        if exact: return [exact[0]]
        chosen = min(langs, key=lambda x: (abs(int(x.get("epoch_number", -10**9)) - target_epoch), x.get("epoch_number", 10**9) > target_epoch))
        return [chosen]

    def _experiment_plot_family(name):
        lname = str(name).lower()
        if "voc_pen" in lname:
            return "voc_pen"
        if "beth_reaper" in lname:
            return "beth_reaper"
        if "voc_len" in lname:
            return "voc_len"
        if "complexity_memory" in lname:
            return "complexity_memory"
        return "default"

    # By default (no explicit mode flags), run both latest and best-accuracy analyses.
    if args.latest_only:
        selection_modes = ["latest"]
    elif args.best_accuracy:
        selection_modes = ["best_accuracy"]
    else:
        selection_modes = ["latest", "best_accuracy"]

    # If a threshold is set, run both: unfiltered + filtered.
    thresholds = [None]
    if args.min_eval_accuracy is not None:
        thresholds = [None, float(args.min_eval_accuracy)]

    for thr in thresholds:
        if thr is None:
            filtered_datapoints = datapoints_list
            experiment_tag = experiment_path
        else:
            before = len(datapoints_list)
            filtered_datapoints = [dp for dp in datapoints_list if np.isfinite(_max_eval_accuracy(dp)) and _max_eval_accuracy(dp) >= thr]
            after = len(filtered_datapoints)
            print(f"Filtered by accuracy >= {thr:.3f}: kept {after}/{before} runs.")
            thr_tag = f"{thr:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            experiment_tag = f"{experiment_path}_accge{thr_tag}"

        for selection_mode in selection_modes:
            prepared_datapoints = []
            for d in filtered_datapoints:
                d2 = dict(d)
                d2["predicates"] = None
                langs = d.get("languages") or []
                if selection_mode == "latest":
                    d2["languages"] = [max(langs, key=lambda x: x.get("epoch_number", -1))] if langs else []
                else:
                    best_ep = _best_accuracy_epoch(d.get("evaluation"))
                    d2["languages"] = _select_language_for_epoch(langs, best_ep) if langs else []
                prepared_datapoints.append(d2)
            gc.collect()

            mode_suffix = "_latest" if selection_mode == "latest" else "_best_accuracy"
            plot_mode_suffix = "latest" if selection_mode == "latest" else "best_accuracy"
            print(f"[INFO] Running variant: experiment={experiment_tag}, mode={plot_mode_suffix}")

            neg_plot_df = None
            neg_plot_df_top1 = None
            if args.negation:
                neg_plot_df, neg_plot_df_top1 = negation.export_analysis(
                    prepared_datapoints,
                    experiment_name=experiment_tag,
                    top_rows=args.negation_top_rows,
                    profile=args.negation_profile,
                    operator=args.feat_operator,
                    latest_only=(selection_mode == "latest"),
                    best_accuracy_only=(selection_mode == "best_accuracy"),
                    n_jobs=args.n_jobs,
                    outputs_dir=pathlib.Path(__file__).resolve().parent / "outputs",
                )

            if args.plots:
                plot_name = f"{experiment_tag}_{plot_mode_suffix}"
                cm = plots.ComplexityMemory(prepared_datapoints, name=plot_name)
                neg_plot_name = f"{experiment_tag}_{plot_mode_suffix}_{args.feat_operator}_{args.negation_profile}"
                cm_neg = plots.ComplexityMemory(prepared_datapoints, name=neg_plot_name)
                plot_family = _experiment_plot_family(experiment_path)

                out_dir = pathlib.Path(__file__).resolve().parent / "plots"
                epochs_df = cm._build_epochs_df()
                if epochs_df is not None:
                    cm.plot_epochs_to_max(epochs_df, out_dir=out_dir)
                else:
                    print("[INFO] Skipping convergence plot (insufficient evaluation data).")
                cfgs = [d.get("config", {}) for d in prepared_datapoints]
                unique_props = {str(c.get("properties")) for c in cfgs if c.get("properties") is not None}
                unique_hidden = {c.get("hidden_size") for c in cfgs if c.get("hidden_size") is not None}
                if plot_family == "complexity_memory":
                    if len(unique_props) > 1:
                        cm.plot_message_compression(group_by=("properties",), out_dir=out_dir)
                    else:
                        print("[INFO] Skipping message_compression by properties (no variation).")
                    if len(unique_hidden) > 1:
                        cm.plot_message_compression(group_by=("hidden_size",), out_dir=out_dir)
                    else:
                        print("[INFO] Skipping message_compression by hidden_size (no variation).")
                else:
                    print(f"[INFO] Skipping message-efficiency plots for family={plot_family}.")

                if neg_plot_df is None:
                    top_rows_csv = cache_dir / f"negation_top_rows_{experiment_tag}_{args.feat_operator}_{args.negation_profile}{mode_suffix}.csv"
                    top1_csv = cache_dir / f"negation_top_1_{experiment_tag}_{args.feat_operator}_{args.negation_profile}{mode_suffix}.csv"
                    if top_rows_csv.is_file():
                        neg_plot_df = pd.read_csv(top_rows_csv)
                        print(f"Loaded existing negation top-rows for plots: {top_rows_csv}")
                    elif top1_csv.is_file():
                        neg_plot_df = pd.read_csv(top1_csv)
                        print(f"Loaded existing negation top-1 for plots (fallback for all negation plots): {top1_csv}")

                if neg_plot_df_top1 is None:
                    top1_csv = cache_dir / f"negation_top_1_{experiment_tag}_{args.feat_operator}_{args.negation_profile}{mode_suffix}.csv"
                    if top1_csv.is_file():
                        neg_plot_df_top1 = pd.read_csv(top1_csv)
                        print(f"Loaded existing negation top-1 for reaper plots: {top1_csv}")

                if neg_plot_df is not None and not neg_plot_df.empty:
                    if plot_family != "voc_pen":
                        cm_neg.plot_negation_metrics_by_complexity(neg_plot_df, out_dir=out_dir)
                    else:
                        print("[INFO] Skipping 'negation by complexity' for family=voc_pen.")

                    if plot_family == "complexity_memory":
                        cm_neg.plot_negation_metrics_over_epochs(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                        if neg_plot_df_top1 is not None and not neg_plot_df_top1.empty:
                            cm_neg.plot_negation_metrics_by_reaper_interval(neg_plot_df_top1, profile_tag=args.feat_operator, out_dir=out_dir)
                        else:
                            print("[WARN] Reaper-interval plots skipped: strict top-1 dataframe is required but not available.")
                        cm_neg.plot_negation_metric_slopes(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                        cm_neg.plot_topsim_vs_negation(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                        cm_neg.plot_topsim_interaction_n(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                    elif plot_family == "beth_reaper":
                        if neg_plot_df_top1 is not None and not neg_plot_df_top1.empty:
                            cm_neg.plot_negation_metrics_by_reaper_interval(
                                neg_plot_df_top1,
                                profile_tag=args.feat_operator,
                                out_dir=out_dir,
                                metrics=["n", "f1_nT"],
                            )
                        else:
                            print("[WARN] Reaper-interval plots skipped: strict top-1 dataframe is required but not available.")
                    elif plot_family == "voc_len":
                        cm_neg.plot_negation_metrics_over_epochs(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                        cm_neg.plot_negation_metric_slopes(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                        cm_neg.plot_topsim_vs_negation(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                        cm_neg.plot_topsim_interaction_n(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                    elif plot_family == "voc_pen":
                        cm_neg.plot_negation_metrics_over_epochs(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                        cm_neg.plot_negation_metric_slopes(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                    else:
                        cm_neg.plot_negation_metrics_over_epochs(neg_plot_df, profile_tag=args.feat_operator, out_dir=out_dir)
                else:
                    print(f"[INFO] No negation dataframe available for negation-specific plots.")
    # cm.plot_message_compression(group_by=("negation","properties"))
