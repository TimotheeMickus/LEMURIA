import os, pathlib
from load import get_datapoints

if __name__ == "__main__":
    # Ask for experimental data
    # Assumes that all runs for one experiment are written into a dedicated folder
    # And that that folder is inside "runs/"
    print(f"Available experiments: {os.listdir(pathlib.Path('runs'))}")
    experiment_path = input("Experiment name: ")
    datapoints = get_datapoints(experiment_path)

    print(f"Found {len(datapoints)} datapoints.")

    for d in datapoints:
        print(d['evaluation'])
        break