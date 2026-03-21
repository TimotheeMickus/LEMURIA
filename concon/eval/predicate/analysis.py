import os, pathlib
from load import get_datapoints

if __name__ == "__main__":
    # Ask for experimental data
    # Assumes that all runs for one experiment are written into a dedicated folder
    # And that that folder is inside "runs/"
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    runs_dir = repo_root / "runs"
    print(f"Available experiments: {os.listdir(runs_dir)}")
    experiment_path = input("Experiment name: ")
    datapoints = get_datapoints(experiment_path)

    # ----- DEBUG / TESTING ONLY -----

    print(f"Found {len(datapoints)} datapoints.")

    has_eval, has_pred, has_lang = 0, 0, 0
    print(f"Data structure:")
    print(datapoints[0].keys())
    for k, v in datapoints[0].items():
        print(f"{k}: {type(v)}")
    for d in datapoints:
        if d['evaluation'] is not None:
            has_eval += 1
        if d['predicates'] is not None:
            has_pred += 1
        if d['languages']:
            has_lang += 1
    print(f"Of which {has_eval} have eval, {has_pred} have pred, {has_lang} have lang.")
