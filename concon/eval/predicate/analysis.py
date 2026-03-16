import os, pathlib
from load import get_datapoints

if __name__ == "__main__":
    # Ask for experimental data
    # Assumes that all runs for one experiment are written into a dedicated folder
    # And that that folder is inside "runs/"
    print(f"Available experiments: {os.listdir(pathlib.Path('runs'))}")
    experiment_path = input("Experiment name: ")
    datapoints = get_datapoints(experiment_path)

    # ----- DEBUG / TESTING ONLY -----

    print(f"Found {len(datapoints)} datapoints.")

    has_eval, has_pred, has_lang = 0, 0, 0
    for d in datapoints:
        if d['evaluation'] is not None:
            has_eval += 1
        if d['predicates'] is not None:
            has_pred += 1
        if d['languages']:
            has_lang += 1
    print(f"Of which {has_eval} have eval, {has_pred} have pred, {has_lang} have lang.")