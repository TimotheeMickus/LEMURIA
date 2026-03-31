import os, pathlib, sys
import argparse
import pickle
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
    print(f"Average vocab length: {vocab_sum/vocab_count}")

    if args.negation:
        pass

    run_plots = args.plots or (args.negation is None)
    cm = plots.ComplexityMemory(datapoints_list, name=experiment_path)
    if run_plots:
        # Plot: epochs to max accuracy/performance
        cm.plot_time_to_max_accuracy()
        if args.negation_mode:
            cm.plot_message_compression(group_by=("negation",))
        cm.plot_message_compression(group_by=("properties",))
        cm.plot_message_compression(group_by=("hidden_size",))
    # cm.plot_message_compression(group_by=("negation","properties"))
