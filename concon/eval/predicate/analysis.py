import os, pathlib
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
    parser.add_argument(
        "--negation-profile",
        choices=["fast", "slow", "unbounded"],
        default="fast",
        help="search profile for negation feature growth (unbounded removes search caps; can be very slow)",
    )
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

    def _load_cached_datapoints(cache_file, experiment_name):
        if cache_file.is_file():
            print(f"Loading cached datapoints from {cache_file} ...")
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load cache ({e}). Re-loading datapoints.")
        print("Loading datapoints...")
        loaded = load.get_datapoints(experiment_name)
        with open(cache_file, "wb") as f:
            pickle.dump(loaded, f)
        print(f"Saved datapoint cache to {cache_file}")
        return loaded

    datapoints_list = _load_cached_datapoints(cache_path, experiment_path)
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
    full_run_names = {str(dp.get("run_name", "")) for dp in datapoints_list}

    for thr in thresholds:
        same_runs_as_unfiltered = False
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
            filtered_run_names = {str(dp.get("run_name", "")) for dp in filtered_datapoints}
            same_runs_as_unfiltered = (filtered_run_names == full_run_names)
            if same_runs_as_unfiltered:
                print("[INFO] Thresholded run set is identical to unfiltered run set.")
        if len(filtered_datapoints) == 0:
            print(f"[INFO] No runs to analyze for experiment={experiment_tag}; skipping this threshold variant.")
            continue

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
            if len(prepared_datapoints) == 0:
                print(f"[INFO] No prepared datapoints for experiment={experiment_tag}, mode={selection_mode}; skipping.")
                continue

            mode_suffix = "_latest" if selection_mode == "latest" else "_best_accuracy"
            plot_mode_suffix = "latest" if selection_mode == "latest" else "best_accuracy"
            is_primary_mode = (selection_mode == selection_modes[0])
            print(f"[INFO] Running variant: experiment={experiment_tag}, mode={plot_mode_suffix}")

            neg_plot_df = None
            neg_plot_df_top1 = None
            if args.negation:
                has_any_language = any((d.get("languages") or []) for d in prepared_datapoints)
                has_any_negation = any(not d.get("config", {}).get("no_negation", False) for d in prepared_datapoints)
                if not has_any_language:
                    print(f"[INFO] Negation export skipped for experiment={experiment_tag}, mode={plot_mode_suffix} (no language dumps).")
                elif not has_any_negation:
                    print(f"[INFO] Negation export skipped for experiment={experiment_tag}, mode={plot_mode_suffix} (all selected runs have no_negation=True).")
                else:
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
                static_plot_name = f"{experiment_tag}"
                cm_static = plots.ComplexityMemory(prepared_datapoints, name=static_plot_name)
                neg_plot_name = f"{experiment_tag}_{plot_mode_suffix}_{args.feat_operator}_{args.negation_profile}"
                plot_family = _experiment_plot_family(experiment_path)
                skip_fast_negation_plots = False
                if args.negation_profile == "fast":
                    fast_rows_csv = cache_dir / f"negation_top_rows_{experiment_tag}_{args.feat_operator}_fast{mode_suffix}.csv"
                    slow_rows_csv = cache_dir / f"negation_top_rows_{experiment_tag}_{args.feat_operator}_slow{mode_suffix}.csv"
                    if fast_rows_csv.is_file() and slow_rows_csv.is_file():
                        skip_fast_negation_plots = True
                        print(f"[INFO] Skipping fast negation plots for {experiment_tag}{mode_suffix} (slow variant is available).")

                out_dir = pathlib.Path(__file__).resolve().parent / "plots"
                if is_primary_mode and not same_runs_as_unfiltered:
                    epochs_df = cm_static._build_epochs_df()
                    if epochs_df is not None:
                        if plot_family == "voc_pen":
                            cm_static.plot_epochs_to_max_by_voc_penalty(epochs_df, out_dir=out_dir)
                        else:
                            cm_static.plot_epochs_to_max(epochs_df, out_dir=out_dir)
                            if plot_family == "beth_reaper":
                                cm_static.plot_epochs_to_max_by_reaper_interval(epochs_df, out_dir=out_dir)
                                cm_static.plot_epochs_to_max_by_reaper_interval_per_complexity(epochs_df, out_dir=out_dir)
                    else:
                        print("[INFO] Skipping convergence plot (insufficient evaluation data).")

                    if plot_family == "beth_reaper":
                        cm_static.plot_eval_accuracy_over_epochs(group_col="reaper_interval", out_dir=out_dir)
                        cm_static.plot_eval_accuracy_over_epochs_by_reaper_and_complexity(out_dir=out_dir)
                    elif plot_family == "voc_pen":
                        cm_static.plot_eval_accuracy_over_epochs(group_col="voc_penalty", out_dir=out_dir)

                    cfgs = [d.get("config", {}) for d in prepared_datapoints]
                    unique_props = {str(c.get("properties")) for c in cfgs if c.get("properties") is not None}
                    unique_hidden = {c.get("hidden_size") for c in cfgs if c.get("hidden_size") is not None}
                    if plot_family == "complexity_memory":
                        if len(unique_props) > 1:
                            cm_static.plot_message_compression(group_by=("properties",), out_dir=out_dir)
                        else:
                            print("[INFO] Skipping message_compression by properties (no variation).")
                        if len(unique_hidden) > 1:
                            cm_static.plot_message_compression(group_by=("hidden_size",), out_dir=out_dir)
                        else:
                            print("[INFO] Skipping message_compression by hidden_size (no variation).")
                    else:
                        print(f"[INFO] Skipping message-efficiency plots for family={plot_family}.")
                elif is_primary_mode and same_runs_as_unfiltered:
                    print("[INFO] Skipping eval/convergence plots for thresholded variant (identical run set as unfiltered).")
                else:
                    print("[INFO] Skipping eval/convergence plots for non-primary mode (identical across mode variants).")

                if not skip_fast_negation_plots:
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

                    neg_plot_sources = []
                    if neg_plot_df is not None and not neg_plot_df.empty:
                        neg_plot_sources.append(("topk", neg_plot_df))
                    if neg_plot_df_top1 is not None and not neg_plot_df_top1.empty:
                        neg_plot_sources.append(("top1", neg_plot_df_top1))

                    if not neg_plot_sources:
                        print(f"[INFO] No negation dataframe available for negation-specific plots.")
                    else:
                        for source_tag, source_df in neg_plot_sources:
                            cm_neg_src = plots.ComplexityMemory(prepared_datapoints, name=f"{neg_plot_name}_{source_tag}")

                            if plot_family != "voc_pen":
                                cm_neg_src.plot_negation_metrics_by_complexity(source_df, out_dir=out_dir)
                            else:
                                print("[INFO] Skipping 'negation by complexity' for family=voc_pen.")

                            if plot_family == "complexity_memory":
                                cm_neg_src.plot_negation_metrics_over_epochs(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                                cm_neg_src.plot_negation_metrics_by_reaper_interval(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                                cm_neg_src.plot_topsim_vs_negation(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                                cm_neg_src.plot_topsim_interaction_n(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                            elif plot_family == "beth_reaper":
                                cm_neg_src.plot_negation_metrics_by_reaper_interval(
                                    source_df,
                                    profile_tag=args.feat_operator,
                                    out_dir=out_dir,
                                    metrics=["n", "f1_nT"],
                                )
                                cm_neg_src.plot_reaper_step_vs_f_scores(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                                cm_neg_src.plot_topsim_vs_n_f1(source_df, control_col="reaper_interval", profile_tag=args.feat_operator, out_dir=out_dir)
                            elif plot_family == "voc_len":
                                cm_neg_src.plot_negation_metrics_over_epochs(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                                cm_neg_src.plot_topsim_vs_negation(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                                cm_neg_src.plot_topsim_interaction_n(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                            elif plot_family == "voc_pen":
                                cm_neg_src.plot_negation_metrics_by_voc_penalty(source_df, out_dir=out_dir)
                                cm_neg_src.plot_f_scores_by_voc_penalty(source_df, out_dir=out_dir)
                                cm_neg_src.plot_negation_metrics_over_epochs(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
                                cm_neg_src.plot_topsim_vs_n_f1(source_df, control_col="voc_penalty", profile_tag=args.feat_operator, out_dir=out_dir)
                            else:
                                cm_neg_src.plot_negation_metrics_over_epochs(source_df, profile_tag=args.feat_operator, out_dir=out_dir)
    # cm.plot_message_compression(group_by=("negation","properties"))
