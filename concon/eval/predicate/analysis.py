import os, pathlib
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
    experiment_path = input("Experiment name: ")

    # ----- DEBUG / TESTING ONLY -----

    has_eval, has_pred, has_lang = 0, 0, 0
    total = 0
    first = None
    print("Loading datapoints...")
    datapoints_list = load.get_datapoints(experiment_path)
    for d in datapoints_list:
        total += 1
        if first is None:
            first = d
        if d['evaluation'] is not None:
            has_eval += 1
        if d['predicates'] is not None:
            has_pred += 1
        if d['languages']:
            has_lang += 1
    print(f"Found {total} datapoints.")
    if first is not None:
        print(f"Data structure:")
        print(first.keys())
        for k, v in first.items():
            print(f"{k}: {type(v)}")
    print(f"Of which {has_eval} have eval, {has_pred} have pred, {has_lang} have lang.")
    print()

    # Plot: epochs to max accuracy/performance
    # plots.ComplexityMemory(datapoints_list, name=experiment_path).plot_time_to_max_accuracy()

    for d in load.iter_datapoints(experiment_path):
        if not d['languages']:
            print(f"no languages found in {d['config']}\n")
        else:
            languages = d["languages"]
            for lang in languages:
                if len(lang['language']['msg'].unique()) > 1:
                    _, vocab = negation._tokenize_messages(lang["language"])
                    max_size = min(6, max(2, len(vocab) // 5))  # e.g. 20% of vocab, capped
                    print(negation.find_negation(languages[0]["language"], min_purity=0, min_coverage=0, mode='greedy'))
                    print()
                    break
