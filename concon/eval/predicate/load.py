import os, pathlib
import json
import pandas as pd

def get_datapoints(directory_name: str = None):
    '''
    Input:
    - directory_name (string): folder found in 'concon/runs'
    Output:
    - datapoints (list[dict]) of runs with keys: 
        - config (dict), 
        - evaluation (DataFrame), 
        - predicates (DataFrame),
        - languages (list[dict[epoch_number: int, DataFrame]])
    '''
    datapoints = []
    super_directory = pathlib.Path('runs') / directory_name

    for directory in super_directory.iterdir():
        # Handle cases where directories are nested
        if not (directory / "hparams.json").is_file():
            subdirectories = [p for p in directory.iterdir() if p.is_dir()]
            if len(subdirectories) == 1 and (subdirectories[0] / "hparams.json").is_file():
                directory = subdirectories[0]
            else:
                print(f"Found weird directory in {super_directory}/{directory}.\n{subdirectories}.")
        # Depending on the run, we have several message dumps: store all
        messages_paths = []
        eval_path = None
        pred_path = None

        # Pick up different CSVs: signals, evaluation, per-predicate metrics.
        for filename in os.listdir(directory):
            if filename.startswith('msgs'):
                messages_paths.append(os.path.join(directory, filename))
            if filename.startswith('eval'):
                eval_path = os.path.join(directory, filename)
            if filename.startswith('predicate'):
                pred_path = os.path.join(directory, filename)

        # Skip runs missing CSV summaries
        if (eval_path is None) or (pred_path is None):
            print(f"[WARN] {directory} at least one csv summary is missing. Skipping...")
            continue

        datapoint = {}
        with open(os.path.join(directory, 'hparams.json')) as f:
            datapoint['config'] = json.load(f)

        datapoint['evaluation'] = pd.read_csv(eval_path)
        datapoint['predicates'] = pd.read_csv(pred_path)
        datapoint['languages'] = []
        for msg_path in messages_paths:
            datapoint['languages'].append({'epoch_number': int(msg_path.split('.')[1].split('e')[1]),
                                           'language': pd.read_csv(msg_path)})

        datapoints.append(datapoint)

    return datapoints
